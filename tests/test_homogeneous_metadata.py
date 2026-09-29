import copy
import unittest
from verl_distill.data.qwen_image21 import canonical_hash
from verl_distill.engine.homogeneous_metadata import validate_metadata, validate_resume

class MetadataValidation(unittest.TestCase):
    def setUp(self):
        self.rows=[dict(id='a',kind='t2i',width=32,height=32,reference_images=[]),dict(id='b',kind='t2i',width=32,height=32,reference_images=[])]
        self.ms=[dict(id=r['id'],kind='t2i',width=32,height=32,reference_count=0,target_tokens=4,reference_tokens=0,text_tokens=2,total_tokens=6) for r in self.rows]
        self.doc=dict(schema=1,records_sha256=canonical_hash(self.rows),cache_index_sha256='cache',metadata=self.ms,metadata_sha256=canonical_hash(self.ms))
    def test_actual_benchmark_configs_and_homogeneous_variants(self):
        import yaml
        from pathlib import Path
        from verl_distill.models.qwen_image21.configuration import validate_qwen_config
        root = Path(__file__).resolve().parents[2]
        for name in ('bench8.yaml', 'bench32.yaml'):
            config = yaml.safe_load((Path(__file__).resolve().parents[1] / 'configs/recipes/qwen_image21/resume200_singlefake_fa3.yaml').read_text())
            validate_qwen_config(config)
            config['runtime'].update(data_ordering='homogeneous_v2', token_metadata_path='/test/metadata.json')
            validate_qwen_config(config)
            for key in ('max_token_ratio', 'max_resolution_ratio'):
                bad = copy.deepcopy(config)
                bad['runtime'][key] = float('nan')
                with self.assertRaises(ValueError): validate_qwen_config(bad)

    def test_valid(self): self.assertEqual(validate_metadata(self.doc,self.rows,'cache'),self.ms)
    def test_each_identity(self):
        for key,bad in [('schema',2),('records_sha256','x'),('cache_index_sha256','x'),('metadata_sha256','x')]:
            with self.subTest(key=key),self.assertRaises(ValueError): validate_metadata({**self.doc,key:bad},self.rows,'cache')
    def test_order_even_if_rehashed(self):
        wrong=list(reversed(self.ms))
        with self.assertRaisesRegex(ValueError,'IDs/order'): validate_metadata({**self.doc,'metadata':wrong,'metadata_sha256':canonical_hash(wrong)},self.rows,'cache')
    def test_length(self):
        with self.assertRaisesRegex(ValueError,'length'): validate_metadata({**self.doc,'metadata':[]},self.rows,'cache')
    def test_geometry(self):
        wrong=copy.deepcopy(self.ms);wrong[0]['target_tokens']=8
        with self.assertRaisesRegex(ValueError,'geometry'): validate_metadata({**self.doc,'metadata':wrong,'metadata_sha256':canonical_hash(wrong)},self.rows,'cache')
    def test_legacy_resume_refused(self):
        for ordering in ('random','bucketed_v1'):
            with self.assertRaisesRegex(ValueError,'explicit migration'):validate_resume({'contract':{'runtime':{'data_ordering':ordering}}})
        good={'contract':{'runtime':{'data_ordering':'homogeneous_v2'}}}
        validate_resume(good)
        with self.assertRaisesRegex(ValueError,'cannot reset'):validate_resume(good,True)


if __name__ == '__main__': unittest.main()
