import unittest
import norch
from norch.utils import utils_unittests as utils
import torch
import sys
import os
import torch.nn.functional as F
from torch.ao.nn.quantized import *
from torch.ao.nn.quantized import functional as qF

class TestTensorOperations(unittest.TestCase):

    def setUp(self):
        self.device = os.environ.get('device')
        if self.device is None or self.device != 'cuda':
            self.device = 'cpu'

        print(f"Running tests on: {self.device}")

    def test_creation_and_conversion(self):
        """
        Test creation and convertion of norch tensor to pytorch
        """
        norch_tensor = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]) 
        torch_tensor = utils.to_torch(norch_tensor)
        self.assertTrue(torch.is_tensor(torch_tensor))

    def test_addition(self):
        """
        Test addition two tensors: tensor1 + tensor2
        """
        norch_tensor1 = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]) 
        norch_tensor2 = norch.Tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]) 
        norch_result = norch_tensor1 + norch_tensor2
        torch_result = utils.to_torch(norch_result) 

        torch_tensor1 = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]) 
        torch_tensor2 = torch.tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]) 
        torch_expected = torch_tensor1 + torch_tensor2

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_addition_broadcasted(self):
        """
        Test addition of two tensors with broadcasting: tensor1 + tensor2
        """
        norch_tensor1 = norch.Tensor([[[1, 2, 3], [4, 5, 6]]])   # Shape (1, 2, 3)
        norch_tensor2 = norch.Tensor([1, 1, 1])   # Shape (3)
        norch_result = norch_tensor1 + norch_tensor2
        torch_result = utils.to_torch(norch_result) 

        torch_tensor1 = torch.tensor([[[1, 2, 3], [4, 5, 6]]])   # Shape (1, 2, 3)
        torch_tensor2 = torch.tensor([1, 1, 1])   # Shape (3)
        torch_expected = torch_tensor1 + torch_tensor2

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

        norch_tensor1 = norch.Tensor([[[10, 10], [-4, -4]], [[5., 6], [7, 8]]])   # Shape (1, 2, 3)
        norch_tensor2 = norch.Tensor([[10, 10], [5, 6]])   # Shape (3)
        norch_result = norch_tensor1 + norch_tensor2
        torch_result = utils.to_torch(norch_result) 
        
        torch_tensor1 = torch.tensor([[[10, 10], [-4, -4]], [[5., 6], [7, 8]]])   # Shape (1, 2, 3)
        torch_tensor2 = torch.tensor([[[10, 10], [5, 6]]])   # Shape (3)
        torch_expected = torch_tensor1 + torch_tensor2

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

        # reversed order broadcasting
        norch_tensor1 = norch.Tensor([[0, 2]])  
        norch_tensor2 = norch.Tensor([[3, 4], [5, -1]])  
        norch_result = norch_tensor1 + norch_tensor2
        torch_result = utils.to_torch(norch_result) 

        torch_tensor1 = torch.tensor([[0, 2]])   
        torch_tensor2 = torch.tensor([[3, 4], [5, -1]])  
        torch_expected = torch_tensor1 + torch_tensor2

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

        norch_result = norch_tensor2 + norch_tensor1
        torch_expected = torch_tensor2 + torch_tensor1

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_subtraction(self):
        """
        Test subtraction of two tensors: tensor1 - tensor2
        """
        norch_tensor1 = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]) 
        norch_tensor2 = norch.Tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]) 
        norch_result = norch_tensor1 - norch_tensor2
        torch_result = utils.to_torch(norch_result) 

        torch_tensor1 = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]) 
        torch_tensor2 = torch.tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]) 
        torch_expected = torch_tensor1 - torch_tensor2

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_broadcasting_subtraction(self):
        """
        Test subtraction of two tensors with broadcasting: tensor1 - tensor2
        """
        norch_tensor1 = norch.Tensor([[[1, 2, 3], [4, 5, 6]]])   # Shape (1, 2, 3)
        norch_tensor2 = norch.Tensor([1, 1, 1])   # Shape (3)
        norch_result = norch_tensor1 - norch_tensor2
        torch_result = utils.to_torch(norch_result) 

        torch_tensor1 = torch.tensor([[[1, 2, 3], [4, 5, 6]]])   # Shape (1, 2, 3)
        torch_tensor2 = torch.tensor([1, 1, 1])   # Shape (3)
        torch_expected = torch_tensor1 - torch_tensor2 

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

        # reversed order broadcasting
        norch_result = norch_tensor2 - norch_tensor1
        torch_result = utils.to_torch(norch_result) 

        torch_expected = torch_tensor2 - torch_tensor1

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_division_by_scalar(self):
        """
        Test division of a tensor by a scalar: tensor / scalar
        """
        norch_tensor = norch.Tensor([[[2, 4], [6, -8]], [[10, 12], [14, 16]]]) 
        scalar = 2
        norch_result = norch_tensor / scalar
        torch_result = utils.to_torch(norch_result) 

        torch_tensor = torch.tensor([[[2, 4], [6, -8]], [[10, 12], [14, 16]]]) 
        torch_expected = torch_tensor / scalar

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_scalar_division_by_tensor(self):
        """
        Test scalar division by a tensor: scalar / tensor
        """
        scalar = 10
        norch_tensor = norch.Tensor([[[2, 4], [6, -8]], [[10, 12], [14, 16]]]) 
        norch_result = scalar / norch_tensor
        torch_result = utils.to_torch(norch_result) 

        torch_tensor = torch.tensor([[[2, 4], [6, -8]], [[10, 12], [14, 16]]]) 
        torch_expected = scalar / torch_tensor

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_matrix_multiplication(self):
        """
        Test matrix multiplication: tensor1 @ tensor2
        """
        norch_tensor1 = norch.Tensor([[[1., 2], [3, 4]], [[5, 6], [7, 8]]]) 
        norch_tensor2 = norch.Tensor([[[1., 0], [0, 1]], [[-1, 0], [0, -1]]]) 
        norch_result = norch_tensor1 @ norch_tensor2
        torch_result = utils.to_torch(norch_result) 

        torch_tensor1 = torch.tensor([[[1., 2], [3, 4]], [[5, 6], [7, 8]]]) 
        torch_tensor2 = torch.tensor([[[1., 0], [0, 1]], [[-1, 0], [0, -1]]]) 
        torch_expected = torch_tensor1 @ torch_tensor2

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_elementwise_multiplication_by_scalar(self):
        """
        Test elementwise multiplication of a tensor by a scalar: tensor * scalar
        """
        norch_tensor = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]) 
        scalar = 2
        norch_result = norch_tensor * scalar
        torch_result = utils.to_torch(norch_result) 

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]) 
        torch_expected = torch_tensor * scalar

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_elementwise_multiplication_by_tensor(self):
        """
        Test elementwise multiplication of two tensors: tensor1 * tensor2
        """
        norch_tensor1 = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]) 
        norch_tensor2 = norch.Tensor([[[2, 2], [2, 2]], [[2, 2], [2, 2]]]) 
        norch_result = norch_tensor1 * norch_tensor2
        torch_result = utils.to_torch(norch_result) 

        torch_tensor1 = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]) 
        torch_tensor2 = torch.tensor([[[2, 2], [2, 2]], [[2, 2], [2, 2]]]) 
        torch_expected = torch_tensor1 * torch_tensor2

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_reshape(self):
        """
        Test reshaping of a tensor: tensor.reshape(shape)
        """
        norch_tensor = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]) 
        new_shape = [2, 4]
        norch_result = norch_tensor.reshape(new_shape)
        torch_result = utils.to_torch(norch_result) 

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]) 
        torch_expected = torch_tensor.reshape(new_shape)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_unsqueeze(self):
        """
        Test unsqueeze operation on a tensor
        """
        norch_tensor = norch.Tensor([[1, 2], [3, 4]]) 
        
        # Unsqueeze at dim=0
        norch_unsqueeze_0 = norch_tensor.unsqueeze(0)
        torch_unsqueeze_0 = utils.to_torch(norch_unsqueeze_0) 
        torch_tensor = torch.tensor([[1, 2], [3, 4]]) 
        torch_expected_0 = torch_tensor.unsqueeze(0)
        self.assertTrue(utils.compare_torch(torch_unsqueeze_0, torch_expected_0))

        # Unsqueeze at dim=1
        norch_unsqueeze_1 = norch_tensor.unsqueeze(1)
        torch_unsqueeze_1 = utils.to_torch(norch_unsqueeze_1) 
        torch_expected_1 = torch_tensor.unsqueeze(1)
        self.assertTrue(utils.compare_torch(torch_unsqueeze_1, torch_expected_1))

        # Unsqueeze at dim=2
        norch_unsqueeze_2 = norch_tensor.unsqueeze(2)
        torch_unsqueeze_2 = utils.to_torch(norch_unsqueeze_2) 
        torch_expected_2 = torch_tensor.unsqueeze(2)
        self.assertTrue(utils.compare_torch(torch_unsqueeze_2, torch_expected_2))

        # Unsqueeze at dim=-1
        norch_unsqueeze_neg_1 = norch_tensor.unsqueeze(-1)
        torch_unsqueeze_neg_1 = utils.to_torch(norch_unsqueeze_neg_1) 
        torch_expected_neg_1 = torch_tensor.unsqueeze(-1)
        self.assertTrue(utils.compare_torch(torch_unsqueeze_neg_1, torch_expected_neg_1))

        # Unsqueeze at dim=-2
        norch_unsqueeze_neg_2 = norch_tensor.unsqueeze(-2)
        torch_unsqueeze_neg_2 = utils.to_torch(norch_unsqueeze_neg_2) 
        torch_expected_neg_2 = torch_tensor.unsqueeze(-2)
        self.assertTrue(utils.compare_torch(torch_unsqueeze_neg_2, torch_expected_neg_2))

    def test_squeeze(self):
        """
        Test squeeze operation on a tensor
        """
        # Create a tensor with some dimensions of size 1
        norch_tensor = norch.Tensor([[[1, 2], [3, 4]]])   # shape [1, 2, 2]
        
        # Squeeze at dim=0
        norch_squeeze_0 = norch_tensor.squeeze(0)
        torch_squeeze_0 = utils.to_torch(norch_squeeze_0) 
        torch_tensor = torch.tensor([[[1, 2], [3, 4]]]) 
        torch_expected_0 = torch_tensor.squeeze(0)
        self.assertTrue(utils.compare_torch(torch_squeeze_0, torch_expected_0))

        # Create a tensor with a dimension of size 1 in the middle
        norch_tensor_middle_1 = norch.Tensor([[[1, 2]], [[3, 4]]])   # shape [2, 1, 2]
        
        # Squeeze at dim=1
        norch_squeeze_1 = norch_tensor_middle_1.squeeze(1)
        torch_squeeze_1 = utils.to_torch(norch_squeeze_1) 
        torch_tensor_middle_1 = torch.tensor([[[1, 2]], [[3, 4]]]) 
        torch_expected_1 = torch_tensor_middle_1.squeeze(1)
        self.assertTrue(utils.compare_torch(torch_squeeze_1, torch_expected_1))

        # Squeeze at dim=-2 (same as dim=1 in this case)
        norch_squeeze_neg_2 = norch_tensor_middle_1.squeeze(-2)
        torch_squeeze_neg_2 = utils.to_torch(norch_squeeze_neg_2) 
        torch_expected_neg_2 = torch_tensor_middle_1.squeeze(-2)
        self.assertTrue(utils.compare_torch(torch_squeeze_neg_2, torch_expected_neg_2))

        # Squeeze all dimensions of size 1 (None)
        norch_tensor_all_1 = norch.Tensor([[[[1, 2], [3, 4]]]])   # shape [1, 1, 2, 2]
        norch_squeeze_all = norch_tensor_all_1.squeeze()
        torch_squeeze_all = utils.to_torch(norch_squeeze_all) 
        torch_tensor_all_1 = torch.tensor([[[[1, 2], [3, 4]]]]) 
        torch_expected_all = torch_tensor_all_1.squeeze()
        self.assertTrue(utils.compare_torch(torch_squeeze_all, torch_expected_all))

        # Squeeze no dimensions (no dimensions of size 1)
        norch_tensor_no_1 = norch.Tensor([[1, 2], [3, 4]])   # shape [2, 2]
        norch_squeeze_none = norch_tensor_no_1.squeeze()
        torch_squeeze_none = utils.to_torch(norch_squeeze_none) 
        torch_tensor_no_1 = torch.tensor([[1, 2], [3, 4]]) 
        torch_expected_none = torch_tensor_no_1.squeeze()
        self.assertTrue(utils.compare_torch(torch_squeeze_none, torch_expected_none))


    def test_transpose(self):
        """
        Test transposition of a tensor: tensor.transpose(dim1, dim2)
        """
        norch_tensor = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]) 
        dim1, dim2 = 0, 2
        norch_result = norch_tensor.transpose(dim1, dim2)
        torch_result = utils.to_torch(norch_result) 

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]) 
        torch_expected = torch_tensor.transpose(dim1, dim2)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_logarithm(self):
        """
        Test elementwise logarithm of a tensor: tensor.log()
        """
        norch_tensor = norch.Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]) 
        norch_result = norch_tensor.log()
        torch_result = utils.to_torch(norch_result) 

        torch_tensor = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]) 
        torch_expected = torch.log(torch_tensor)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_sum(self):
        """
        Test summation of a tensor: tensor.sum()
        """
        norch_tensor = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]) 
        norch_result = norch_tensor.sum()
        torch_result = utils.to_torch(norch_result) 

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]) 
        torch_expected = torch.sum(torch_tensor)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_sum_axis(self):
        """
        Test summation of a tensor along a specific axis without keeping the dimensions
        """
        norch_tensor = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]) 
        norch_result = norch_tensor.sum(axis=1)
        torch_result = utils.to_torch(norch_result) 

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]) 
        torch_expected = torch.sum(torch_tensor, dim=1)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

        # negative axis

        norch_tensor = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]) 
        norch_result = norch_tensor.sum(axis=-2)
        torch_result = utils.to_torch(norch_result) 

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]) 
        torch_expected = torch.sum(torch_tensor, dim=-2)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))


    def test_sum_axis_keepdim(self):
        """
        Test summation of a tensor along a specific axis with keepdim=True
        """
        norch_tensor = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]) 
        norch_result = norch_tensor.sum(axis=1, keepdim=True)
        torch_result = utils.to_torch(norch_result) 

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]) 
        torch_expected = torch.sum(torch_tensor, dim=1, keepdim=True)
        
        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_max(self):
        """
        Test max of a tensor: tensor.max()
        """
        norch_tensor = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]) 
        norch_result = norch_tensor.max()
        torch_result = utils.to_torch(norch_result) 

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]) 
        torch_expected = torch.max(torch_tensor)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_max_axis(self):
        """
        Test max of a tensor along a specific axis without keeping the dimensions
        """
        norch_tensor = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]) 
        norch_result = norch_tensor.max(axis=1)
        torch_result = utils.to_torch(norch_result) 

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]) 
        torch_expected, _ = torch.max(torch_tensor, dim=1)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

        # negative axis

        norch_tensor = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]) 
        norch_result = norch_tensor.max(axis=-1)
        torch_result = utils.to_torch(norch_result) 

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]) 
        torch_expected, _ = torch.max(torch_tensor, dim=-1)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))


    def test_max_axis_keepdim(self):
        """
        Test max of a tensor along a specific axis with keepdim=True
        """
        norch_tensor = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]) 
        norch_result = norch_tensor.max(axis=1, keepdim=True)
        torch_result = utils.to_torch(norch_result) 

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]) 
        torch_expected, _ = torch.max(torch_tensor, dim=1, keepdim=True)
        
        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_min(self):
        """
        Test min of a tensor: tensor.min()
        """
        norch_tensor = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]) 
        norch_result = norch_tensor.min()
        torch_result = utils.to_torch(norch_result) 

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]) 
        torch_expected = torch.min(torch_tensor)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_min_axis(self):
        """
        Test min of a tensor along a specific axis without keeping the dimensions
        """
        norch_tensor = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]) 
        norch_result = norch_tensor.min(axis=1)
        torch_result = utils.to_torch(norch_result) 

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]) 
        torch_expected, _ = torch.min(torch_tensor, dim=1)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

        # negative axis

        norch_tensor = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]) 
        norch_result = norch_tensor.min(axis=-1)
        torch_result = utils.to_torch(norch_result) 

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]) 
        torch_expected, _ = torch.min(torch_tensor, dim=-1)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))


    def test_min_axis_keepdim(self):
        """
        Test min of a tensor along a specific axis with keepdim=True
        """
        norch_tensor = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]) 
        norch_result = norch_tensor.min(axis=1, keepdim=True)
        torch_result = utils.to_torch(norch_result) 

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]) 
        torch_expected, _ = torch.min(torch_tensor, dim=1, keepdim=True)
        
        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_1D_T(self):
        """
        Test transposition of a 1D tensor.
        """
        norch_tensor = norch.Tensor([1, 2, 3, 4]) 
        norch_result = norch_tensor.T
        torch_result = utils.to_torch(norch_result) 

        torch_tensor = torch.tensor([1, 2, 3, 4]) 
        torch_expected = torch_tensor.T

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_2D_T(self):
        """
        Test transposition of a 2D tensor.
        """
        norch_tensor = norch.Tensor([[1, 2, 3], [4, 5, 6]]) 
        norch_result = norch_tensor.T
        torch_result = utils.to_torch(norch_result) 

        torch_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]]) 
        torch_expected = torch_tensor.T

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_3D_T(self):
        """
        Test transposition of a tensor: tensor.T
        """
        norch_tensor = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]) 
        norch_result = norch_tensor.T
        torch_result = utils.to_torch(norch_result) 

        torch_tensor = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]) 
        torch_expected = torch_tensor.T

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_matmul(self):
        """
        Test matrix multiplication: MxP = NxM @ MxP
        """
        # Creating batched tensors for Norch
        norch_tensor1 = norch.Tensor([[1., 2], [3, -4], [5, 6], [7, 8]]) 
        norch_tensor2 = norch.Tensor([[2., 3, 1, 0, 4], [5, -1, 2, 3, 0]]) 

        norch_result = norch_tensor1 @ norch_tensor2
        torch_result = utils.to_torch(norch_result) 

        # Converting to PyTorch tensors for comparison
        torch_tensor1 = torch.tensor([[1., 2], [3, -4], [5, 6], [7, 8]]) 
        torch_tensor2 = torch.tensor([[2., 3, 1, 0, 4], [5, -1, 2, 3, 0]]) 

        torch_expected = torch.matmul(torch_tensor1, torch_tensor2)

        # Comparing results
        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_reshape_then_matmul(self):
        """
        Test reshaping a tensor followed by matrix multiplication: (tensor.reshape(shape) @ other_tensor)
        """
        norch_tensor = norch.Tensor([[1., 2], [3, -4], [5, 6], [7, 8]]) 
        new_shape = [2, 4]
        norch_reshaped = norch_tensor.reshape(new_shape)
        
        norch_result = norch_reshaped @ norch_tensor
        torch_result = utils.to_torch(norch_result) 

        torch_tensor = torch.tensor([[1., 2], [3, -4], [5, 6], [7, 8]]) 
        torch_expected = torch_tensor.reshape(new_shape) @ torch_tensor

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_batched_matmul(self):
        """
        Test batched matrix multiplication: BxMxP = BxNxM @ BxMxP
        """
        B = 3  # Batch size

        # Creating batched tensors for Norch
        norch_tensor1 = norch.Tensor([[[1., 2], [3, -4], [5, 6], [7, 8]] for _ in range(B)]) 
        norch_tensor2 = norch.Tensor([[[2., 3, 1, 0, 4], [5, -1, 2, 3, 0]] for _ in range(B)]) 

        norch_result = norch_tensor1 @ norch_tensor2
        torch_result = utils.to_torch(norch_result) 

        # Converting to PyTorch tensors for comparison
        torch_tensor1 = torch.tensor([[[1., 2], [3, -4], [5, 6], [7, 8]] for _ in range(B)]) 
        torch_tensor2 = torch.tensor([[[2., 3, 1, 0, 4], [5, -1, 2, 3, 0]] for _ in range(B)]) 

        torch_expected = torch.matmul(torch_tensor1, torch_tensor2)

        # Comparing results
        self.assertTrue(utils.compare_torch(torch_result, torch_expected))


    def test_broadcasted_batched_matmul(self):
        """
        Test broadcasted batched matrix multiplication: BxMxP = NxM @ BxMxP
        """
        B = 3  # Batch size

        # Creating batched tensors for Norch
        norch_tensor1 = norch.Tensor([[1., 2], [3, -4], [5, 6], [7, 8]]) 
        norch_tensor2 = norch.Tensor([[[2., 3, 1, 0, 4], [5, -1, 2, 3, 0]] for _ in range(B)]) 

        norch_result = norch_tensor1 @ norch_tensor2
        torch_result = utils.to_torch(norch_result) 

        # Converting to PyTorch tensors for comparison
        torch_tensor1 = torch.tensor([[1., 2], [3, -4], [5, 6], [7, 8]]) 
        torch_tensor2 = torch.tensor([[[2., 3, 1, 0, 4], [5, -1, 2, 3, 0]] for _ in range(B)]) 

        torch_expected = torch.matmul(torch_tensor1, torch_tensor2)

        # Comparing results
        self.assertTrue(utils.compare_torch(torch_result, torch_expected))



    def test_transpose_then_matmul(self):
        """
        Test transposing a tensor followed by matrix multiplication: (tensor.transpose(dim1, dim2) @ other_tensor)
        """
        norch_tensor = norch.Tensor([[[1., 2], [3, 4]], [[5, 6], [7, 8]]]) 
        dim1, dim2 = 0, 2
        norch_result = norch_tensor.transpose(dim1, dim2) @ norch_tensor
        torch_result = utils.to_torch(norch_result) 

        torch_tensor = torch.tensor([[[1., 2], [3, 4]], [[5, 6], [7, 8]]]) 
        torch_expected = torch_tensor.transpose(dim1, dim2) @ torch_tensor

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_add_div_matmul_then_reshape(self):
        """
        Test a combination of operations: (tensor.sum() + other_tensor) / scalar @ another_tensor followed by reshape
        """
        norch_tensor1 = norch.Tensor([[[1., 2], [3, -4]], [[5, 6], [7, 8]]]) 
        norch_tensor2 = norch.Tensor([[[1., 1], [1, 1]], [[1, 1], [1, 1]]]) 
        scalar = 2
        new_shape = [2, 4]
        norch_result = ((norch_tensor1 + norch_tensor2) / scalar) @ norch_tensor1
        norch_result = norch_result.reshape(new_shape)
        torch_result = utils.to_torch(norch_result) 

        torch_tensor1 = torch.tensor([[[1., 2], [3, -4]], [[5, 6], [7, 8]]]) 
        torch_tensor2 = torch.tensor([[[1., 1], [1, 1]], [[1, 1], [1, 1]]]) 
        torch_expected = ((torch_tensor1 + torch_tensor2) / scalar) @ torch_tensor1
        torch_expected = torch_expected.reshape(new_shape)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_scalar_power_tensor(self):
        """
        Test scalar power of a tensor: scalar ** tensor
        """
        scalar = 3
        norch_tensor = norch.Tensor([[[1, 2.1], [3, -4]], [[5, 6], [7, 8]]]) 
        norch_result = scalar ** norch_tensor
        torch_result = utils.to_torch(norch_result) 

        torch_tensor = torch.tensor([[[1, 2.1], [3, -4]], [[5, 6], [7, 8]]]) 
        torch_expected = scalar ** torch_tensor

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_tensor_power_scalar(self):
        """
        Test tensor power of a scalar: tensor ** scalar
        """
        scalar = 3
        norch_tensor = norch.Tensor([[[1, 2.1], [3, -4]], [[5, 6], [7, 8]]]) 
        norch_result = norch_tensor ** scalar
        torch_result = utils.to_torch(norch_result) 

        torch_tensor = torch.tensor([[[1, 2.1], [3, -4]], [[5, 6], [7, 8]]]) 
        torch_expected = torch_tensor ** scalar

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_tensor_sin(self):
        """
        Test sine function on tensor
        """
        norch_tensor = norch.Tensor([[[0, 30], [45, 60]], [[90, 120], [135, 180]]]) 
        norch_result = norch_tensor.sin()
        torch_result = utils.to_torch(norch_result) 

        torch_tensor = torch.tensor([[[0, 30], [45, 60]], [[90, 120], [135, 180]]]) 
        torch_expected = torch.sin(torch_tensor)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_tensor_cos(self):
        """
        Test cosine function on tensor
        """
        norch_tensor = norch.Tensor([[[0, 30], [45, 60]], [[90, 120], [135, 180]]]) 
        norch_result = norch_tensor.cos()
        torch_result = utils.to_torch(norch_result) 

        torch_tensor = torch.tensor([[[0, 30], [45, 60]], [[90, 120], [135, 180]]]) 
        torch_expected = torch.cos(torch_tensor)

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_equal(self):
        """
        Test equal two tensors: tensor1.equal(tensor2)
        """
        norch_tensor1 = norch.Tensor([[[1, 2], [3, -4]], [[5, 1], [7, 8]]]) 
        norch_tensor2 = norch.Tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]) 
        norch_result = norch_tensor1.equal(norch_tensor2)
        torch_result = utils.to_torch(norch_result) 

        torch_tensor1 = torch.tensor([[[1, 2], [3, -4]], [[5, 1], [7, 8]]]) 
        torch_tensor2 = torch.tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]) 
        torch_expected = (torch_tensor1 == torch_tensor2).float()

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

    def test_broadcasted_equal(self):
        """
        Test broadcasted equal two tensors: tensor1.equal(tensor2)
        """
        norch_tensor1 = norch.Tensor([[[10, 10], [-4, -4]], [[5., 6], [7, 8]]]) 
        norch_tensor2 = norch.Tensor([[[10, 10]], [[5, 6]]]) 
        norch_result = norch_tensor1.equal(norch_tensor2)
        torch_result = utils.to_torch(norch_result) 

        torch_tensor1 = torch.tensor([[[10, 10], [-4, -4]], [[5., 6], [7, 8]]]) 
        torch_tensor2 = torch.tensor([[[10, 10]], [[5, 6]]]) 
        torch_expected = (torch_tensor1 == torch_tensor2).float()

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))

        norch_tensor1 = norch.Tensor([[[10, 10], [-4, -4]], [[5., 6], [7, 8]]]) 
        norch_tensor2 = norch.Tensor([[[10.0,], [-4.0,]],[[6.0,], [8.0,]]]) 
        norch_result = norch_tensor1.equal(norch_tensor2)
        torch_result = utils.to_torch(norch_result) 

        torch_tensor1 = torch.tensor([[[10, 10], [-4, -4]], [[5., 6], [7, 8]]]) 
        torch_tensor2 = torch.tensor([[[10.0,], [-4.0,]],[[6.0,], [8.0,]]]) 
        torch_expected = (torch_tensor1 == torch_tensor2).float()

        self.assertTrue(utils.compare_torch(torch_result, torch_expected))
    

    def test_zeros_like(self):
        """
        Test creating a tensor of zeros with the same shape as another tensor.
        """
        norch_tensor = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]) 
        norch_zeros = norch_tensor.zeros_like()
        torch_zeros_result = utils.to_torch(norch_zeros) 

        torch_tensor_expected = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]) 
        torch_zeros_expected = torch.zeros_like(torch_tensor_expected)

        self.assertTrue(utils.compare_torch(torch_zeros_result, torch_zeros_expected))

    def test_ones_like(self):
        """
        Test creating a tensor of ones with the same shape as another tensor.
        """
        norch_tensor = norch.Tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]) 
        norch_ones = norch_tensor.ones_like()
        torch_ones_result = utils.to_torch(norch_ones) 

        torch_tensor_expected = torch.tensor([[[1, 2], [3, -4]], [[5, 6], [7, 8]]]) 
        torch_ones_expected = torch.ones_like(torch_tensor_expected)

        self.assertTrue(utils.compare_torch(torch_ones_result, torch_ones_expected))

    def test_conv2d(self):
        """
        Test 2D convolution operation with bias
        """
        print("Testing conv2d with bias")
        for _ in range(100):
            input = torch.randn(1, 3, 32, 32)  # input∼N(0,1) 
            weight = torch.randn(16, 3, 3, 3)  # weight∼N(0,1)
            bias = torch.randn(16)  # bias∼N(0,1)
            
            norch_input = norch.Tensor(input.tolist())
            norch_weight = norch.Tensor(weight.tolist())
            norch_bias = norch.Tensor(bias.tolist())
            
            norch_result = norch_input.conv2d(norch_weight, bias=norch_bias, stride=2, padding=1)
            torch_expected = torch.nn.functional.conv2d(input, weight, bias=bias, stride=2, padding=1)
            
            # print("PyTorch result shape:", torch_expected.shape)
            # print("PyTorch result:", torch_expected[0][0:2][0][0:2])
            # print("Norch result:", utils.to_torch(norch_result)[0][0:2][0][0:2])
            torch.testing.assert_close(utils.to_torch(norch_result), torch_expected)

    def test_qconv2d(self):
        """
        Test quantized 2D convolution operation
        """
        print("Testing qconv2d")
        filters = torch.randn(8, 4, 3, 3, dtype=torch.float)
        inputs = torch.randn(1, 4, 3, 3, dtype=torch.float)
        bias = torch.randn(8, dtype=torch.float)

        # Asymmetric quantization for input
        input_scale = (inputs.max() - inputs.min()) / 255
        input_zero_point = (-inputs.min() / input_scale).round().clamp(0, 255).to(dtype=torch.int32)
        q_inputs = torch.quantize_per_tensor(inputs, input_scale.item(), input_zero_point, torch.quint8)

        # Symmetric quantization for filter
        filter_scale = (filters.abs().max())*2 / 255
        q_filters = torch.quantize_per_tensor(filters, filter_scale.item(), 0, torch.qint8)
        
        # Estimate output range, Calculate output scale and zero point
        conv_output = F.conv2d(inputs, filters, bias, padding=1)
        output_scale = (conv_output.max() - conv_output.min()) / 255
        output_zero_point = (-conv_output.min() / output_scale).round().clamp(0, 255).to(dtype=torch.int32)


        # multiplier = input_scale*filter_scale/output_scale
        # q_output = qF.conv2d(q_inputs, q_filters, bias, padding=1, scale=multiplier, zero_point=output_zero_point.item())
        q_output = qF.conv2d(q_inputs, q_filters, bias, padding=1, scale=output_scale.item(), zero_point=output_zero_point.item())
        print(q_inputs.int_repr())
        print(q_filters.int_repr())
        qbias = torch.round(bias / (input_scale * filter_scale))

        # print(q_output.int_repr())
        # print(q_output.shape)

if __name__ == '__main__':
    unittest.main()
