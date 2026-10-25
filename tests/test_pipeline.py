import logging
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from cifar100.config import load_config, ROOT
from cifar100.models import build_model
from cifar100.models.swin import head_optimizer, finetune_optimizer
from cifar100.data import get_split, validate_split
from cifar100.checkpoint import load_checkpoint
from cifar100.engine import evaluate
from cifar100.pipeline import run_stage
from cifar100.reporting import save_evaluation
from cifar100.utils import create_run

torch.set_num_threads(2)


class PipelineTests(unittest.TestCase):
    def test_split_reuse_and_invalid_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = dict(seed=42, validation_size=200, split_path=str(Path(tmp)/'split.npz'))
            labels = np.repeat(np.arange(100), 10)
            a,b = get_split(labels,config)
            c,d = get_split(labels,config)
            np.testing.assert_array_equal(a,c)
            np.testing.assert_array_equal(b,d)
            self.assertTrue(np.all(np.bincount(labels[b]) == 2))
            with self.assertRaises(ValueError):
                get_split(labels,{**config,'seed':43})
            with self.assertRaises(ValueError):
                validate_split([0,1],[1,3],4,2)

    def test_resnet_mixup_checkpoint_and_reports(self):
        config = load_config('configs/resnet_mixup.json')
        self.assertTrue((ROOT/'train.py').exists())
        model = build_model(config)
        self.assertEqual(sum(p.numel() for p in model.parameters()),1096196)
        inputs = torch.randn(4,3,32,32)
        labels = torch.tensor([0,1,2,3])
        loader = DataLoader(TensorDataset(inputs,labels),batch_size=2)
        before = model.classifier[2].weight.detach().clone()
        with tempfile.TemporaryDirectory() as tmp:
            config.update(results_dir=tmp+'/results',checkpoint_dir=tmp+'/ckpt')
            output,ckpt = create_run(config,'test')
            optimizer = torch.optim.SGD(model.parameters(),lr=.01)
            history=[]
            best = run_stage(model,(loader,loader),torch.nn.CrossEntropyLoss(label_smoothing=.1),optimizer,None,
                torch.device('cpu'),config,[str(i) for i in range(100)],([0,1],[2,3]),'train',1,ckpt,output,history,logging.getLogger('test'))
            self.assertFalse(torch.equal(before,model.classifier[2].weight))
            restored,metadata = load_checkpoint(best,torch.device('cpu'))
            model.eval(); restored.eval()
            torch.testing.assert_close(model(inputs),restored(inputs))
            bn_before=restored.stem[1].running_mean.clone()
            result=evaluate(restored,loader,torch.nn.CrossEntropyLoss(),torch.device('cpu'),collect=True)
            torch.testing.assert_close(bn_before,restored.stem[1].running_mean)
            self.assertEqual(result['samples'],4)
            summary=save_evaluation(result,output/'evaluation',[str(i) for i in range(100)],config,{'split':'synthetic'})
            self.assertTrue((output/'evaluation/confusion_counts.png').exists())
            self.assertTrue((output/'figures/learning_curves.png').exists())
            self.assertEqual(summary['samples'],4)
            with self.assertRaises(FileExistsError):
                create_run(config,'test')

    def test_swin_head_and_finetune(self):
        config = load_config('configs/swin_t.json')
        model = build_model(config,pretrained=False)
        self.assertEqual(sum(p.numel() for p in model.parameters()),27596254)
        opt=head_optimizer(model,config)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),76900)
        from cifar100.engine import train_one_epoch
        loader=DataLoader(TensorDataset(torch.randn(2,3,224,224),torch.tensor([0,1])),batch_size=2)
        train_one_epoch(model,loader,torch.nn.CrossEntropyLoss(),opt,torch.device('cpu'))
        opt=finetune_optimizer(model,config)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertEqual([g['lr'] for g in opt.param_groups],[config['backbone_lr'],config['finetune_head_lr']])
        train_one_epoch(model,loader,torch.nn.CrossEntropyLoss(),opt,torch.device('cpu'))
        self.assertEqual(model(torch.randn(1,3,224,224)).shape,(1,100))


if __name__=='__main__':
    unittest.main()
