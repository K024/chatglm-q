import torch
import unittest
from chatglm_q.triton_ops import dynamic_quant_matmul, dynamic_quant_matmul_transposed


class TritonOpsTests(unittest.TestCase):

    def test_dynamic_quant_matmul(self):
        A = torch.randn((10, 128)).cuda()
        B = torch.randint(-127, 127, (128, 256), dtype=torch.int8).cuda()
        B_scale = torch.randn((256, )).cuda() / 256

        result = dynamic_quant_matmul(A, B, B_scale, allow_tf32=False)

        expected = A @ (B * B_scale)
        self.assertTrue(torch.allclose(result, expected, atol=1e-4, rtol=1e-4))


    def test_dynamic_quant_matmul_transposed(self):
        A = torch.randn((10, 128)).cuda()
        B = torch.randint(-127, 127, (256, 128), dtype=torch.int8).cuda()
        B_scale = torch.randn((128, )).cuda() / 256

        result = dynamic_quant_matmul_transposed(A, B, B_scale, allow_tf32=False)

        stable = A @ (B * B_scale).t()
        self.assertTrue(torch.allclose(result, stable, atol=1e-4, rtol=1e-4))


if __name__ == '__main__':
    unittest.main()
