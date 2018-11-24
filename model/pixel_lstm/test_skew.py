from skewers import skew, unskew
import torch
from random import randint
from functools import reduce

def test_identity(dims, debug=False):
    print("Testing skew/unskew identity")
    array = torch.arange(reduce(lambda x,y:x*y, dims), dtype=torch.float32).reshape(dims)
    if debug:
        print("Input:")
        print(array)
    skewed = skew(array)
    if debug:
        print("Skewed:")
        print(skewed)
    unskewed = unskew(skewed)
    if debug:
        print("Unskewed:")
        print(unskewed)
    assert torch.all(array == unskewed).data, "Skew/Unskew Test FAILED."
    print("Skew/Unskew Test PASSED")

def test_skew(debug=False):
    print("Testing skew")
    array = torch.FloatTensor([[[[1,2,3],[4,5,6]]]])
    if debug:
        print("Input:")
        print(array)
    skewed = skew(array)
    correct = torch.FloatTensor([[[[1,2,3,0],[0,4,5,6]]]])
    if debug:
        print("Skewed:")
        print(skewed)
    assert torch.all(skewed == correct).data, "Skew Test FAILED"
    print("Skew test PASSED")

def test_unskew(debug=False):
    print("Testing unskew")
    array = torch.FloatTensor([[[[1,2,3,0],[0,4,5,6]]]])
    if debug:
        print("Input:")
        print(array)
    unskewed = unskew(array)
    correct = torch.FloatTensor([[[[1,2,3],[4,5,6]]]])
    if debug:
        print("Unskewed:")
        print(unskewed)
    assert torch.all(unskewed == correct).data, "Unskew Test FAILED"
    print("Unskew test PASSED")

def main():
    for i in range(3):
        dims = 4 * [randint(1,101)]
        try:
            test_identity(dims)
        except AssertionError:
            test_identity(dims, debug=True)    
    try:
        test_skew()
    except AssertionError:
        test_skew(debug=True)

    try:
        test_unskew()
    except AssertionError:
        test_unskew(debug=True)
    

if __name__ == "__main__":
    main()
