from typing import List, Dict, Tuple

def get_bin(value, width) -> str: 
  return format(value, 'b').zfill(width)

def GenerateEquivalentClassesImpl(table: Dict[Tuple[int, int], List[str]], sum: int, width: int, symmetry_width: int) -> List[str]:
  if (width == 1):
    return [get_bin(sum, width)]
  if (sum, width) in table:
    return table[(sum, width)]
  result = []
  half_width = width // 2
  for i in range(0, sum // 2+1):
    if (sum - i) > half_width:
      continue
    if i > half_width:
      continue
    if (i == sum - i):
      left_list = GenerateEquivalentClassesImpl(table, i, half_width, symmetry_width)
      for j in range(len(left_list)):
        for k in range(j+1):
          result.append(left_list[k] + left_list[j])
          if width > symmetry_width and left_list[k] != left_list[j]:
            result.append(left_list[j] + left_list[k])
    else:
      left_list = GenerateEquivalentClassesImpl(table, sum-i, half_width, symmetry_width)
      right_list = GenerateEquivalentClassesImpl(table, i, half_width, symmetry_width)
      for left_entry in left_list:
        for right_entry in right_list:
          result.append(left_entry + right_entry)
          if width > symmetry_width and right_entry != left_entry:
            result.append(right_entry + left_entry)
  table[(sum, width)] = result
  return result

def GenerateEquivalentClasses(width: int, symmetry_width: int) -> List[str]:
  result = []
  table = {}
  for i in range(0, width+1):
    result.extend(GenerateEquivalentClassesImpl(table, i, width, symmetry_width))
  return result

def GetEquivalentClassSize(value: str, width: int, symmetry_width: int) -> int:
  if (width == 1):
    return 1
  half_width = width // 2
  left = GetEquivalentClassSize(value[0:half_width], half_width, symmetry_width)
  if (value[0:half_width] != value[half_width:]):
    right = GetEquivalentClassSize(value[half_width:], half_width, symmetry_width)
    if (width <= symmetry_width):
      return left * right * 2
    return left * right
  return left * left
