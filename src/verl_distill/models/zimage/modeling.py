import logging
import types
from contextlib import contextmanager

import torch
import torch.distributed as dist
from diffusers import (
    ZImagePipeline,
    ZImageTransformer2DModel,
)

from .transformer import ZImageTransformer2DModelWrapper

logger = logging.getLogger(__name__)


def _patch_prepare_sequence_pad_token_cast(transformer):
    prepare_sequence = getattr(transformer, "_prepare_sequence", None)
    if prepare_sequence is None or getattr(transformer, "_verl_pad_token_cast_patch", False):
        return
    original = (
        prepare_sequence.__func__ if hasattr(prepare_sequence, "__func__") else prepare_sequence
    )

    def _prepare_sequence_with_cast(
        self, feats, pos_ids, inner_pad_mask, pad_token, *args, **kwargs
    ):
        if feats:
            pad_token = pad_token.to(dtype=feats[0].dtype, device=feats[0].device)
        return original(self, feats, pos_ids, inner_pad_mask, pad_token, *args, **kwargs)

    transformer._prepare_sequence = types.MethodType(_prepare_sequence_with_cast, transformer)
    transformer._verl_pad_token_cast_patch = True


class GenTransformer(torch.nn.Module):
    accepts_aux_time_meta = True

    def __init__(self, transformer, vae_scale_factor, aux_time_embed) -> None:
        super().__init__()
        _patch_prepare_sequence_pad_token_cast(transformer)
        self.transformer = transformer
        self.config = transformer.config
        self.in_channels = transformer.config.in_channels
        self.vae_scale_factor = vae_scale_factor
        self.aux_time_embed = aux_time_embed
        self.accepts_aux_time_meta = True
        self.aux_time_log_path = None
        self.last_aux_time_record = None

    def set_aux_time_log_path(self, aux_time_log_path):
        self.aux_time_log_path = aux_time_log_path

    @staticmethod
    def _is_rank0():
        return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

    def enable_gradient_checkpointing(self):
        self.transformer.enable_gradient_checkpointing()

    def gradient_checkpointing_enable(self, *args, **kwargs):
        def _gradient_checkpointing_func(module, *args):
            return torch.utils.checkpoint.checkpoint(
                module.__call__,
                *args,
                **kwargs["gradient_checkpointing_kwargs"],
            )

        self.transformer.enable_gradient_checkpointing(_gradient_checkpointing_func)

    def init_weights(self):
        pass

    @staticmethod
    def _external_sigma_to_internal(sigma: torch.Tensor) -> torch.Tensor:
        sigma_abs = sigma.abs()
        return torch.where(sigma < 0, -(1.0 - sigma_abs), 1.0 - sigma_abs)

    def init_discriminator_head(self, *args, **kwargs):
        if not hasattr(self.transformer, "init_discriminator_head"):
            raise ValueError("Underlying transformer does not support discriminator head")
        return self.transformer.init_discriminator_head(*args, **kwargs)

    def init_multi_feature_discriminator_head(self, *args, **kwargs):
        if not hasattr(self.transformer, "init_multi_feature_discriminator_head"):
            raise ValueError(
                "Underlying transformer does not support multi-feature discriminator head"
            )
        return self.transformer.init_multi_feature_discriminator_head(*args, **kwargs)

    def init_dual_projector_multi_feature_discriminator_head(self, *args, **kwargs):
        if not hasattr(
            self.transformer,
            "init_dual_projector_multi_feature_discriminator_head",
        ):
            raise ValueError(
                "Underlying transformer does not support dual multi-feature discriminator head"
            )
        return self.transformer.init_dual_projector_multi_feature_discriminator_head(
            *args,
            **kwargs,
        )

    def discriminator_parameters(self):
        if not hasattr(self.transformer, "discriminator_parameters"):
            raise ValueError("Underlying transformer does not support discriminator head")
        yield from self.transformer.discriminator_parameters()

    def multi_feature_discriminator_parameters(self):
        if not hasattr(self.transformer, "multi_feature_discriminator_parameters"):
            raise ValueError(
                "Underlying transformer does not support multi-feature discriminator head"
            )
        yield from self.transformer.multi_feature_discriminator_parameters()

    def dual_projector_multi_feature_discriminator_parameters(self):
        if not hasattr(
            self.transformer,
            "dual_projector_multi_feature_discriminator_parameters",
        ):
            raise ValueError(
                "Underlying transformer does not support dual multi-feature discriminator head"
            )
        yield from self.transformer.dual_projector_multi_feature_discriminator_parameters()

    def add_adapter(self, *args, **kwargs):
        self.transformer.add_adapter(*args, **kwargs)

    def set_adapter(self, *args, **kwargs):
        self.transformer.set_adapter(*args, **kwargs)

    def disable_adapter(self, *args, **kwargs):
        if hasattr(self.transformer, "disable_adapter"):
            adapter_context = self.transformer.disable_adapter(*args, **kwargs)
            if adapter_context is not None:
                return adapter_context
        elif hasattr(self.transformer, "disable_adapters"):
            adapter_context = self.transformer.disable_adapters(*args, **kwargs)
            if adapter_context is not None:
                return adapter_context

        @contextmanager
        def _reenable_adapter():
            try:
                yield
            finally:
                if hasattr(self.transformer, "enable_adapters"):
                    self.transformer.enable_adapters()
                elif hasattr(self.transformer, "enable_adapter"):
                    self.transformer.enable_adapter()
                elif hasattr(self.transformer, "enable_lora"):
                    self.transformer.enable_lora()

        return _reenable_adapter()

    def disable_adapters(self, *args, **kwargs):
        return self.disable_adapter(*args, **kwargs)

    def disable_lora(self):
        self.transformer.disable_lora()

    def enable_lora(self):
        self.transformer.enable_lora()

    def forward(self, x_t, t, c=None, tt=None, **kwargs):
        feature_layers = kwargs.pop("feature_layers", None)
        discriminator_mode = bool(kwargs.pop("discriminator_mode", False))
        discriminator_return_raw = bool(kwargs.pop("return_raw", False))
        skip_aux_time = bool(kwargs.pop("skip_aux_time", False))
        disable_separate_r_modulation = bool(kwargs.pop("disable_separate_r_modulation", False))
        discriminator_output = kwargs.pop("discriminator_output", None)
        discriminator_head = kwargs.pop("discriminator_head", None)
        return_discriminator_features = bool(
            kwargs.pop("return_features", False)
            or kwargs.pop("return_discriminator_features", False)
        )
        if c is None:
            c = kwargs.get("c", None)
        if c is None:
            raise ValueError(
                "Condition 'c' must be provided either as positional or keyword argument"
            )
        if discriminator_mode:
            if not self.aux_time_embed:
                raise ValueError("discriminator_mode requires aux_time_embed=True")
            if tt is None:
                raise ValueError("tt must be provided for discriminator_mode")
        use_aux_time = self.aux_time_embed and tt is not None and not skip_aux_time

        batch_size = x_t.shape[0]

        # Z-Image expects List[Tensor(C, F, H, W)] where F=1 for images.
        x_t_ = x_t.unsqueeze(2)
        x_list = list(x_t_.unbind(dim=0))

        encoder_hs = c[0]
        encoder_hs_mask = c[1]

        txt_seq_lens = encoder_hs_mask.int().sum(dim=1).tolist()
        txt_seq_lens = [int(i) for i in txt_seq_lens]
        max_txt_len = max(txt_seq_lens)

        encoder_hs = encoder_hs[:, :max_txt_len]

        # Z-Image expects List[Tensor(L_i, D)] with actual lengths.
        cap_feats_list = []
        for i in range(batch_size):
            actual_len = txt_seq_lens[i]
            cap_feats_list.append(encoder_hs[i, :actual_len])

        if use_aux_time:
            t_flat = t.reshape(-1)
            tt_flat = tt.reshape(-1)
            if t_flat.numel() != tt_flat.numel():
                if tt_flat.numel() == 1:
                    tt_flat = tt_flat.expand_as(t_flat)
                    tt = tt_flat
                elif t_flat.numel() == 1:
                    t_flat = t_flat.expand_as(tt_flat)
                    t = t_flat
                else:
                    raise ValueError(
                        f"t and tt must have same number of elements or be broadcastable, got {t_flat.numel()} and {tt_flat.numel()}"
                    )

            eq_mask = t_flat == (-tt_flat)
            zero_pair_mask = (t_flat == 0) & (tt_flat == 0)
            adjust_mask = eq_mask & zero_pair_mask

            if adjust_mask.any():
                t = t.clone()
                tt = tt.clone()
                t_view = t.reshape(-1)
                tt_view = tt.reshape(-1)
                t_view[adjust_mask] = 0.0001
                tt_view[adjust_mask] = -0.0001

            aux_time_meta = kwargs.get("aux_time_meta", None)
            has_aux_time_meta = isinstance(aux_time_meta, dict)
            aux_time_meta = aux_time_meta if has_aux_time_meta else {}
            step = int(aux_time_meta.get("step", -1))
            call_tag = str(aux_time_meta.get("call_tag", "main"))
            sample_roles = aux_time_meta.get("sample_roles", None)

            t_record = t.reshape(-1).detach().to(torch.float32)
            tt_record = tt.reshape(-1).detach().to(torch.float32)
            eq_record = eq_mask.reshape(-1).detach()
            adjust_record = adjust_mask.reshape(-1).detach()

            self.last_aux_time_record = {
                "step": step,
                "call_tag": call_tag,
                "t": t_record,
                "tt": tt_record,
                "eq_mask": eq_record,
                "adjust_mask": adjust_record,
            }
            if sample_roles is not None:
                self.last_aux_time_record["sample_roles"] = sample_roles

            if has_aux_time_meta and self.aux_time_log_path is not None and self._is_rank0():
                t_list = t_record.cpu().tolist()
                tt_list = tt_record.cpu().tolist()
                eq_list = eq_record.cpu().tolist()
                adjust_list = adjust_record.cpu().tolist()

                with open(self.aux_time_log_path, "a", encoding="utf-8") as f:
                    for i, (tv, ttv, eqv, adjv) in enumerate(
                        zip(t_list, tt_list, eq_list, adjust_list)
                    ):
                        roles = []
                        if sample_roles is not None and i < len(sample_roles):
                            role_value = sample_roles[i]
                            if isinstance(role_value, str):
                                roles = [role_value]
                            else:
                                roles = [str(role) for role in role_value]
                        roles_str = "[" + ",".join(roles) + "]"
                        f.write(
                            f"{step}\t{call_tag}\t{i}\t"
                            f"{float(tv):.8f}\t{float(ttv):.8f}\t"
                            f"{int(bool(eqv))}\t{int(bool(adjv))}\t{roles_str}\n"
                        )
        # Z-Image uses internal time coordinates. For positive sigma inputs,
        # the internal value is 1 - sigma. When aux time is provided,
        # downstream TimeRotaryModulator consumes tt in this internal
        # coordinate directly, so tt represents 1 - generator_sigma.
        t = self._external_sigma_to_internal(t)
        transformer_kwargs = {
            "x": x_list,
            "t": t,
            "cap_feats": cap_feats_list,
            "patch_size": 2,
            "f_patch_size": 1,
            "return_dict": False,
        }

        if use_aux_time:
            tt = self._external_sigma_to_internal(tt)
            transformer_kwargs["target_timestep"] = tt
            if disable_separate_r_modulation:
                transformer_kwargs["disable_separate_r_modulation"] = True
        if discriminator_mode:
            transformer_kwargs["discriminator_mode"] = True
            if discriminator_output is not None:
                transformer_kwargs["discriminator_output"] = discriminator_output
            if discriminator_head is not None:
                transformer_kwargs["discriminator_head"] = discriminator_head
            if return_discriminator_features:
                transformer_kwargs["return_discriminator_features"] = True

        if feature_layers is not None:
            transformer_kwargs["feature_layers"] = tuple(feature_layers)
        output = self.transformer(**transformer_kwargs)
        if feature_layers is not None:
            return output

        if discriminator_mode:
            if return_discriminator_features:
                logits, features = output
                if discriminator_return_raw:
                    return logits, features
                return logits.reshape(-1), features
            logits = output[0]
            if discriminator_return_raw:
                return logits
            return logits.reshape(-1)

        output = output[0]
        output_tensor = torch.stack(output, dim=0)
        prediction = output_tensor.squeeze(2)

        return -prediction

    def discriminate(self, x_t, t, c=None, tt=None, **kwargs):
        return self.forward(
            x_t,
            t=t,
            c=c,
            tt=tt,
            discriminator_mode=True,
            **kwargs,
        )

    def forward_with_cfg(
        self,
        x,
        t,
        c=None,
        cfg_scale=None,
        cfg_interval=None,
        tt=None,
        **kwargs,
    ):
        if c is None:
            c = kwargs.get("c", None)
        if cfg_scale is None:
            cfg_scale = kwargs.get("cfg_scale", 1.0)
        if c is None:
            raise ValueError("Condition 'c' must be provided")

        if cfg_interval is None:
            cfg_interval = [0.0, 1.0]

        t = t.flatten()
        if t[0] >= cfg_interval[0] and t[0] <= cfg_interval[1]:
            half = x[: len(x) // 2]
            combined = torch.cat([half, half], dim=0)
            if tt is not None:
                tt = tt.flatten()[: len(combined)]
            model_out = self.forward(combined, t[: len(combined)], c=c, tt=tt)

            eps, rest = (
                model_out[:, : self.in_channels],
                model_out[:, self.in_channels :],
            )
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)

            eps = torch.cat([half_eps, half_eps], dim=0)
            eps = torch.cat([eps, rest], dim=1)
        else:
            half = x[: len(x) // 2]
            t = t[: len(t) // 2]
            c = [c_[: len(c_) // 2] for c_ in c]
            half_eps = self.forward(half, t, c=c, tt=tt)
            eps = torch.cat([half_eps, half_eps], dim=0)

        return eps


class ZImage(torch.nn.Module):
    def __init__(
        self,
        model_id,
        model_type="t2i",
        aux_time_embed=False,
        text_dtype=torch.bfloat16,
        imgs_dtype=torch.bfloat16,
        max_sequence_length=1024,
        device="cuda",
        load_transformer=True,
    ) -> None:
        super().__init__()

        self.aux_time_embed = aux_time_embed
        z_image_transformer = None
        if load_transformer:
            if aux_time_embed:
                transformer_cls = ZImageTransformer2DModelWrapper
            else:
                transformer_cls = ZImageTransformer2DModel

            z_image_transformer = transformer_cls.from_pretrained(
                model_id,
                subfolder="transformer",
                torch_dtype=imgs_dtype,
                low_cpu_mem_usage=False,
            )

        self.model_type = model_type
        if model_type == "t2i":
            self.model = ZImagePipeline.from_pretrained(
                model_id,
                torch_dtype=imgs_dtype,
                transformer=z_image_transformer,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.device = device
        self.max_sequence_length = max_sequence_length

        self.imgs_dtype = imgs_dtype
        self.text_dtype = text_dtype

        if load_transformer:
            # Keep transformer on the same device as sampling latents.
            self.model.transformer = self.model.transformer.to(dtype=self.imgs_dtype).to(
                self.device
            )
            self.transformer = GenTransformer(
                self.model.transformer,
                self.model.vae_scale_factor,
                self.aux_time_embed,
            )
        else:
            self.model.transformer = None
            self.transformer = None

        self.model.vae = (
            self.model.vae.to(dtype=self.imgs_dtype).requires_grad_(False).eval().to(device)
        )
        self.model.text_encoder = (
            self.model.text_encoder.to(dtype=self.text_dtype)
            .requires_grad_(False)
            .eval()
            .to(device)
        )

    def forward(self, x_t, t, c=None, tt=None, **kwargs):
        if self.transformer is None:
            raise ValueError("ZImage transformer was not loaded")
        return self.transformer(x_t, t, c=c, tt=tt, **kwargs)

    def get_no_split_modules(self):
        text_encoder_no_split_modules = [m for m in self.model.text_encoder._no_split_modules]
        transformer_no_split_modules = (
            [m for m in self.model.transformer._no_split_modules]
            if self.model.transformer is not None
            else []
        )
        return text_encoder_no_split_modules + transformer_no_split_modules

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.vae.eval()
        self.model.text_encoder.eval()
        return self

    def eval(self):
        return self.train(False)

    def requires_grad_(self, requires_grad: bool = True):
        if self.transformer is not None:
            self.transformer.requires_grad_(requires_grad)
        return self

    def encode_prompt(self, prompt, image=None, do_cfg=True):
        if do_cfg:
            if self.model_type == "t2i":
                prompt_embeds_list, neg_prompt_embeds_list = self.model.encode_prompt(
                    prompt=prompt,
                    negative_prompt=None,
                    do_classifier_free_guidance=True,
                    device=self.device,
                    max_sequence_length=self.max_sequence_length,
                )

                max_len_pos = max(len(emb) for emb in prompt_embeds_list)
                max_len_neg = max(len(emb) for emb in neg_prompt_embeds_list)
                max_len = max(max_len_pos, max_len_neg)

                batch_size = len(prompt_embeds_list)
                embed_dim = prompt_embeds_list[0].shape[-1]
                device = prompt_embeds_list[0].device
                dtype = prompt_embeds_list[0].dtype

                prompt_embeds = torch.zeros(
                    (batch_size, max_len, embed_dim),
                    device=device,
                    dtype=dtype,
                )
                prompt_attention_mask = torch.zeros(
                    (batch_size, max_len),
                    device=device,
                    dtype=dtype,
                )

                for i, emb in enumerate(prompt_embeds_list):
                    length = len(emb)
                    prompt_embeds[i, :length] = emb
                    prompt_attention_mask[i, :length] = 1.0

                neg_prompt_embeds = torch.zeros(
                    (batch_size, max_len, embed_dim),
                    device=device,
                    dtype=dtype,
                )
                neg_prompt_attention_mask = torch.zeros(
                    (batch_size, max_len),
                    device=device,
                    dtype=dtype,
                )

                for i, emb in enumerate(neg_prompt_embeds_list):
                    length = len(emb)
                    neg_prompt_embeds[i, :length] = emb
                    neg_prompt_attention_mask[i, :length] = 1.0

            elif self.model_type == "edit":
                raise NotImplementedError("Edit mode is not yet fully implemented for Z-Image")

            return (
                prompt_embeds.to(self.imgs_dtype),
                prompt_attention_mask.to(self.imgs_dtype),
                neg_prompt_embeds.to(self.imgs_dtype),
                neg_prompt_attention_mask.to(self.imgs_dtype),
            )
        else:
            if self.model_type == "t2i":
                prompt_embeds_list, _ = self.model.encode_prompt(
                    prompt=prompt,
                    negative_prompt=None,
                    do_classifier_free_guidance=False,
                    device=self.device,
                    max_sequence_length=self.max_sequence_length,
                )

                max_len = max(len(emb) for emb in prompt_embeds_list)
                batch_size = len(prompt_embeds_list)
                embed_dim = prompt_embeds_list[0].shape[-1]
                device = prompt_embeds_list[0].device
                dtype = prompt_embeds_list[0].dtype

                prompt_embeds = torch.zeros(
                    (batch_size, max_len, embed_dim),
                    device=device,
                    dtype=dtype,
                )
                prompt_attention_mask = torch.zeros(
                    (batch_size, max_len),
                    device=device,
                    dtype=dtype,
                )

                for i, emb in enumerate(prompt_embeds_list):
                    length = len(emb)
                    prompt_embeds[i, :length] = emb
                    prompt_attention_mask[i, :length] = 1.0

            elif self.model_type == "edit":
                raise NotImplementedError("Edit mode is not yet fully implemented for Z-Image")

            return (
                prompt_embeds.to(self.imgs_dtype),
                prompt_attention_mask.to(self.imgs_dtype),
                None,
                None,
            )

    @torch.no_grad()
    def pixels_to_latents(self, pixels):
        pixel_values = pixels.to(self.model.vae.dtype)
        pixel_latents = self.model.vae.encode(pixel_values).latent_dist.mean
        pixel_latents = (
            pixel_latents - self.model.vae.config.shift_factor
        ) * self.model.vae.config.scaling_factor

        return pixel_latents

    def latents_to_pixels(self, latents):
        latents = latents.to(self.model.vae.dtype)
        latents = (
            latents / self.model.vae.config.scaling_factor
        ) + self.model.vae.config.shift_factor
        pixels = self.model.vae.decode(latents, return_dict=False)[0]

        return pixels

    @torch.no_grad()
    def sample(
        self,
        prompts,
        images=None,
        cfg_scale=4.5,
        seed=42,
        height=512,
        width=512,
        times=1,
        return_traj=False,
        sampler=None,
        sampler_kwargs=None,
        use_ema=False,
    ):
        do_cfg = cfg_scale > 0.0
        (
            prompt_embeds,
            prompt_attention_mask,
            neg_prompt_embeds,
            neg_prompt_attention_mask,
        ) = self.encode_prompt(prompts, images, do_cfg)

        noise = torch.randn(
            [
                len(prompts) * times,
                self.transformer.in_channels,
                height // self.model.vae_scale_factor,
                width // self.model.vae_scale_factor,
            ],
            dtype=torch.float32,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).to(self.device)

        if do_cfg:
            prompt_embeds = torch.cat(
                (times * [prompt_embeds] + times * [neg_prompt_embeds]), dim=0
            )
            pooled_prompt_embeds = torch.cat(
                (times * [prompt_attention_mask] + times * [neg_prompt_attention_mask]),
                dim=0,
            )
            latents = torch.cat([noise] * 2)
            if use_ema:
                assert hasattr(self, "ema_transformer"), (
                    "`use_ema` is set True but `ema_transformer` is not initialized"
                )
                model_fn = self.ema_transformer.forward_with_cfg
            else:
                model_fn = self.transformer.forward_with_cfg
        else:
            latents = noise
            prompt_embeds = torch.cat(times * [prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat(times * [prompt_attention_mask], dim=0)
            if use_ema:
                assert hasattr(self, "ema_transformer"), (
                    "`use_ema` is set True but `ema_transformer` is not initialized"
                )
                model_fn = self.ema_transformer
            else:
                model_fn = self.transformer

        if do_cfg:
            model_kwargs = dict(
                c=[prompt_embeds, pooled_prompt_embeds],
                cfg_scale=cfg_scale,
                cfg_interval=[0.0, 1.0],
            )
        else:
            model_kwargs = dict(c=[prompt_embeds, pooled_prompt_embeds])

        if sampler_kwargs is None:
            sampler_kwargs = {}

        sampler_call_kwargs = dict(sampler_kwargs)
        sampler_call_kwargs.update(model_kwargs)
        latents = sampler(latents, model_fn, **sampler_call_kwargs)

        if do_cfg:
            cfg_batch_dim = 1 if latents.ndim == 5 else 0
            latents, _ = latents.chunk(2, dim=cfg_batch_dim)

        if latents.ndim == 5:
            if return_traj:
                latents = latents.reshape(-1, *latents.shape[2:])
            else:
                latents = latents[-1]
        else:
            latents = latents.reshape(-1, *latents.shape[1:])
            if not return_traj:
                latents = latents[-len(prompts) * times :]

        if return_traj:
            images = []
            for i in range(len(latents)):
                latent = latents[i : i + 1].to(self.device)
                image = self.latents_to_pixels(latent)
                images.append(image.detach().to(torch.float32).cpu())
            images = torch.cat(images, dim=0)
            return images
        else:
            images = self.latents_to_pixels(latents.to(self.device))

        return images

    @torch.no_grad()
    def prepare_data(
        self,
        prompt,
        images,
        times=1,
    ):
        do_cfg = True
        (
            prompt_embeds,
            prompt_attention_mask,
            neg_prompt_embeds,
            neg_prompt_attention_mask,
        ) = self.encode_prompt(prompt, do_cfg=do_cfg)

        if do_cfg:
            prompt_embeds = torch.cat(
                (times * [prompt_embeds] + times * [neg_prompt_embeds]), dim=0
            )
            pooled_prompt_embeds = torch.cat(
                (times * [prompt_attention_mask] + times * [neg_prompt_attention_mask]),
                dim=0,
            )
        latents = self.pixels_to_latents(images.to(self.device))
        c = (
            prompt_embeds[: times * len(prompt)],
            pooled_prompt_embeds[: times * len(prompt)],
            prompt_embeds[times * len(prompt) :],
            pooled_prompt_embeds[times * len(prompt) :],
        )
        return latents, c
