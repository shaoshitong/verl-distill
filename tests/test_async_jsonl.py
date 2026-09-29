import json,tempfile,unittest,threading
from pathlib import Path
from unittest.mock import patch
from verl_distill.engine.async_jsonl import AsyncJSONL
class AsyncLogs(unittest.TestCase):
 def test_snapshot_order_and_close_drain(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'log';w=AsyncJSONL(capacity=2);r={'step':0,'nested':[0]}
   w.enqueue(p,r);r['nested'][0]=99
   for i in range(1,20):w.enqueue(p,{'step':i})
   w.close();rows=[json.loads(x) for x in p.read_text().splitlines()]
   self.assertEqual([x['step'] for x in rows],list(range(20)))
   self.assertEqual(rows[0]['nested'],[0]);self.assertFalse(w.thread.is_alive())
 def test_io_failure_visible(self):
  with tempfile.TemporaryDirectory() as d:
   w=AsyncJSONL();w.enqueue(Path(d)/'absent'/'log',{'step':1})
   with self.assertRaisesRegex(RuntimeError,'write failed'):w.flush()
   with self.assertRaises(RuntimeError):w.close()
 def test_reject_non_json_before_enqueue(self):
  w=AsyncJSONL()
  with self.assertRaises(TypeError):w.enqueue('unused',{'bad':object()})
  with self.assertRaises(ValueError):w.enqueue('unused',{'bad':float('nan')})
  self.assertEqual(w.queue.unfinished_tasks,0);w.close()
 def test_backpressure_bounded(self):
  gate=threading.Event();entered=threading.Event()
  def stalled(*a,**k):entered.set();gate.wait(2);raise OSError('injected')
  with patch('verl_distill.engine.async_jsonl.Path.open',side_effect=stalled):
   w=AsyncJSONL(capacity=1,timeout=.05);w.enqueue('x',{})
   self.assertTrue(entered.wait(1));w.enqueue('x',{})
   with self.assertRaises(TimeoutError):w.enqueue('x',{})
   gate.set()
   with self.assertRaises(RuntimeError):w.close()
