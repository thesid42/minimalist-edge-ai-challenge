import copy
import json
from pathlib import Path
import tempfile
import unittest
import torch
from torch import nn
from edgeprep.challenge import restore_recipe
from edgeprep.structural import apply_edits, plan_edits


class StructuralReplayTests(unittest.TestCase):
    def test_edited_recovered_checkpoint_replays_shapes_and_exact_weights(self):
        torch.set_num_threads(2)
        parent_path=Path('best_models/resize192_head_ce_w16_e391533162d7/recipe.json')
        if not parent_path.exists():self.skipTest('Recovered parent artifact is required')
        parent,recipe=restore_recipe(parent_path)
        self.assertTrue(Path(recipe["recovered_checkpoint"]).is_absolute())
        self.assertTrue(Path(recipe["recovered_checkpoint"]).exists())
        original_parameters=sum(p.numel() for p in parent.parameters())
        edits=plan_edits(parent,'head640')+plan_edits(parent,'drop16')
        model=apply_edits(parent,edits)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            state={k:v.detach().cpu() for k,v in model.state_dict().items()}
            torch.save(state,folder/'recovered_state.pt')
            from edgeprep.challenge_data import digest
            recipe.update(structural_edits=edits,recovered_checkpoint=str(folder/'recovered_state.pt'),recovered_sha256=digest(folder/'recovered_state.pt'))
            (folder/'recipe.json').write_text(json.dumps(recipe),encoding='utf-8')
            restored,_=restore_recipe(folder/'recipe.json')
            self.assertIsInstance(restored.student.features[16],nn.Identity)
            self.assertEqual(restored.student.classifier[1].in_features,640)
            self.assertLess(sum(p.numel() for p in restored.parameters()),original_parameters)
            for key,value in state.items():torch.testing.assert_close(value,restored.state_dict()[key],rtol=0,atol=0)
            self.assertTrue(all(v.dtype==torch.float32 for v in restored.parameters()))
            with torch.no_grad():
                inputs=torch.randn(1,3,224,224)
                torch.testing.assert_close(model(inputs),restored(inputs),rtol=0,atol=0)
                self.assertEqual(restored(inputs).shape,(1,10))

    def test_block_removal_rejects_shape_changing_block(self):
        torch.set_num_threads(2)
        parent_path=Path('best_models/resize192_head_ce_w16_e391533162d7/recipe.json')
        if not parent_path.exists():self.skipTest('Recovered parent artifact is required')
        model,_=restore_recipe(parent_path)
        with self.assertRaisesRegex(ValueError,'shape-preserving'):
            apply_edits(model,[{'kind':'drop_residual','feature_index':14}])


if __name__=='__main__':unittest.main()
