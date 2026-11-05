"""Target-coverage tests; no checkpoint or dataset required."""
import unittest
from full_validation import full_windows

class CoverageTests(unittest.TestCase):
    def test_all_targets_once_and_context_bounded(self):
        for n in [2,511,512,513,1024,1025,1026,1536,1537,2048,4434606]:
            windows=full_windows(n)
            expected=1
            count=0
            for begin,end,first in windows:
                self.assertEqual(first,expected)
                self.assertLess(begin,first)
                self.assertLessEqual(end-begin,1024)
                self.assertLess(end,n)
                count+=end-first+1
                expected=end+1
            self.assertEqual(count,n-1)
            self.assertEqual(expected,n)
        self.assertEqual(len(full_windows(4434606)),8661)

    def test_first_and_final_partial_window(self):
        self.assertEqual(full_windows(1026),[(0,1024,1),(1,1025,1025)])
        self.assertEqual(full_windows(1),[])

if __name__=='__main__': unittest.main()
