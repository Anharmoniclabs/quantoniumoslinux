"""Check that only the planned half of the supervision changes at every update."""
import unittest
from types import SimpleNamespace
from data_mixture_comparison import schedules, describe
from data_comparison_train import _inspect_schedule
from chat_recovery_data import SOURCES


def example(uid, source, values):
    return dict(id=uid,source=source,input_ids=[7,8]+values,labels=[-100]+values+[2])


class ScheduleTests(unittest.TestCase):
    def test_common_slots_match_and_source_targets_survive_row_boundaries(self):
        pools={s:[example(s,s,[9,10,11,12,13])] for s in SOURCES}
        paired=[dict(**example(f'p-{i}', 'paired_code', [21, 22, 23]),family='code',pair_id='p')
                for i in range(2)]
        a=schedules(pools,paired,3,8)
        for name,rows in a.items():
            _inspect_schedule(rows,40,128,2)
            self.assertEqual(sum(describe(rows)['targets_per_slot'].values()),24)
            self.assertEqual(describe(rows)['targets_per_slot']['common'],12)
        self.assertEqual([r for r in a['paired'] if r['slot']=='common'],
                         [r for r in a['broad'] if r['slot']=='common'])
        sequence=[y for r in a['paired'] if r['slot']=='paired_substitution' for y in r['labels'] if y!=-100]
        self.assertEqual(sequence,[21,22,23,2]*3)
        self.assertEqual(a,schedules(pools,paired,3,8))
