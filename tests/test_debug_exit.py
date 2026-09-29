import unittest
from verl_distill.engine.debug_exit import capture_update,force_debug_exit
class DebugExit(unittest.TestCase):
 def test_debug_pair_and_neighbours(self):
  for boundary in (100,200,300):
   self.assertTrue(capture_update('fake_score',boundary-1,100))
   self.assertTrue(capture_update('generator',boundary,100))
   self.assertFalse(capture_update('fake_score',boundary,100))
   self.assertFalse(capture_update('generator',boundary-5,100))
 def test_disabled_modes(self):
  self.assertFalse(capture_update('reflow',0,100))
  self.assertFalse(capture_update('fake_score',99,100,benchmark=True))
  for capture,enabled,rollout in ((False,True,True),(True,False,True),(True,True,False)):
   self.assertFalse(force_debug_exit(capture,enabled,rollout))
  self.assertTrue(force_debug_exit(True,True,True))
 def test_interval_ten_sequence(self):
  captured=[]
  for f in range(1,31):
   if capture_update('fake_score',f-1,10):captured.append(('fake_score',f))
   if f%5==0 and capture_update('generator',f,10):captured.append(('generator',f))
  self.assertEqual(captured,[(p,f) for f in (10,20,30) for p in ('fake_score','generator')])
