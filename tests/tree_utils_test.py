from snnax.utils.util import apply_and_keep_pytype

# Simple test with list
test_list = list(range(10))
print(apply_and_keep_pytype(lambda x: x + 1, test_list))

# Simple test with list
fl = 3.
print(apply_and_keep_pytype(lambda x: 2.*x, fl))

