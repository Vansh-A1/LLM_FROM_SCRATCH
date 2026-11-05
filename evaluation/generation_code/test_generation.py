import unittest
import sys
import torch
import generation as g

class SamplingTests(unittest.TestCase):
    def test_top_p_keeps_crossing_token_and_one_minimum(self):
        # p=.9 nucleus must retain both 0.7 and 0.25, excluding 0.05.
        logits = torch.tensor([.7, .25, .05]).log() * .8
        out = g.filtered_logits(logits, 'top_p')
        self.assertEqual(torch.isfinite(out).tolist(), [True, True, False])
        out = g.filtered_logits(torch.tensor([100., 0., -1.]), 'top_p')
        self.assertEqual(torch.isfinite(out).tolist(), [True, False, False])

    def test_top_k_exact_cutoff(self):
        out = g.filtered_logits(torch.arange(50.), 'top_k')
        self.assertEqual(torch.isfinite(out).sum().item(), 40)
        self.assertTrue(torch.isneginf(out[:10]).all())

    def test_combined_top_k_then_nucleus(self):
        logits = torch.arange(50.) / 30
        out = g.filtered_logits(logits, 'combined')
        self.assertLess(torch.isfinite(out).sum().item(), 40)
        self.assertTrue(torch.isneginf(out[:10]).all())

    def test_greedy_unchanged(self):
        x = torch.tensor([2., 3., -1.])
        self.assertEqual(g.filtered_logits(x, 'greedy').argmax().item(), 1)

    def test_same_seed_isolated(self):
        before = torch.random.get_rng_state().clone()
        p = g.filtered_logits(torch.linspace(-1, 1, 50), 'combined').softmax(0)
        a = torch.multinomial(p, 20, replacement=True, generator=torch.Generator().manual_seed(42))
        b = torch.multinomial(p, 20, replacement=True, generator=torch.Generator().manual_seed(42))
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_schedule_counts(self):
        self.assertEqual(sum(len(v) for v in g.PROMPTS.values()), 30)
        self.assertEqual([len(v) for v in g.PROMPTS.values()], [8, 6, 6, 5, 5])
        self.assertEqual(30 * sum(len(m['seeds']) for m in g.METHODS.values()), 300)
        self.assertTrue(all(p.endswith('\n    ') for p in g.PROMPTS['code_continuation']))

    def test_final_position_logits_and_context_crop(self):
        sys.path.insert(0, str(g.ROOT))
        from safe_model import architecture
        module = architecture()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(123)
            model = module.GPT(module.GPTConfig(n_layer=2, n_head=2, n_embd=16,
                block_size=8, vocab_size=300, dropout=0., bias=True)).eval().requires_grad_(False)
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        with torch.inference_mode():
            expected = model(tokens[:, -8:])[0][0, -1]
            actual = g.next_logits(model, tokens)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

if __name__ == '__main__':
    unittest.main()
