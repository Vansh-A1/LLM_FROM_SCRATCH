"""Known-answer CPU tests; no learned model load or GPU allocation."""
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
import tiktoken
from lm_eval.api.instance import Instance
from adapter import NanoGPTLM


class PredictNext(torch.nn.Module):
    """At each token x, expected next token is (x+1)%10 with logit 3."""
    def __init__(self,uniform=False):
        super().__init__(); self.weight=torch.nn.Parameter(torch.zeros(1),requires_grad=False)
        self.config=SimpleNamespace(block_size=8); self.uniform=uniform; self.calls=0; self.inputs=[]
        self.eval()
    def forward(self,x):
        self.calls+=1;self.inputs.append(x.tolist())
        y=torch.zeros(*x.shape,10)
        if not self.uniform: y.scatter_(-1,((x+1)%10).unsqueeze(-1),3.0)
        return y,None


class Tests(unittest.TestCase):
    def setUp(self):
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        self.tokenizer=SimpleNamespace(eot_token=9)
        self.identity={"sha256":"synthetic-not-a-model","source_sha256":"synthetic"}
    def adapter(self,model=None,**kwargs):
        return NanoGPTLM(model or PredictNext(),self.tokenizer,self.identity,batch_size=3,**kwargs)
    def test_uniform_known_answer_padding_and_lengths(self):
        a=self.adapter(PredictNext(True))
        out=a._loglikelihood_tokens([(("a","b"),[1,2],[3,4,5]),(("c","d"),[8],[9])],disable_tqdm=True)
        self.assertAlmostEqual(out[0][0],-3*math.log(10),places=5)
        self.assertAlmostEqual(out[1][0],-math.log(10),places=5)
        self.assertFalse(out[0][1])
    def test_causal_shift_and_greedy(self):
        a=self.adapter();out=a._loglikelihood_tokens([(("a","b"),[1,2],[3,4,5]),(("c","d"),[1],[9])],True)
        logp=3-math.log(math.exp(3)+9)
        self.assertAlmostEqual(out[0][0],3*logp,places=5);self.assertTrue(out[0][1]);self.assertFalse(out[1][1])
    def test_truncation_matches_reference(self):
        model=PredictNext();a=self.adapter(model)
        ctx=[1,2,3,4,5,6,7,8,9,0,1,2];cont=[3,4,5]
        a._loglikelihood_tokens([(("a","b"),ctx,cont)],True)
        self.assertEqual(model.inputs[0][0],(ctx+cont)[-9:][:-1])
    def test_cache_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            model=PredictNext();a=self.adapter(model,cache_path=Path(tmp)/"cache.sqlite")
            req=[(("a","b"),[1,2],[3,4,5])]
            first=a._loglikelihood_tokens(req,True);a.close()
            second=self.adapter(model,cache_path=Path(tmp)/"cache.sqlite")
            self.assertEqual(first,second._loglikelihood_tokens(req,True));self.assertEqual(model.calls,1);second.close()
    def test_leading_space_and_empty_context(self):
        a=self.adapter();a.tokenizer=tiktoken.get_encoding("gpt2")
        ctx,cont=a._encode_pair("Hello ","world.")
        self.assertEqual(ctx,a.tok_encode("Hello"));self.assertEqual(cont,a.tok_encode(" world."))
        captured=[]
        def capture(req,**kwargs):captured.extend(req);return [(0.,False)]*len(req)
        a._loglikelihood_tokens=capture
        a.loglikelihood([Instance("loglikelihood",{},("","hello"),0)],disable_tqdm=True)
        self.assertEqual(captured[0][1],[50256]);self.assertEqual(captured[0][2],a.tok_encode("hello"))
    def test_reject_empty_and_too_long_continuation(self):
        a=self.adapter()
        for cont in [[],list(range(9))]:
            with self.assertRaises(ValueError): a._loglikelihood_tokens([(("a","b"),[1],cont)],True)
    def test_reject_train_mode(self):
        model=PredictNext();model.train()
        with self.assertRaises(ValueError): self.adapter(model)

if __name__=="__main__":unittest.main()
