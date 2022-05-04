import numpy as np

def calculate(list):
  try:
    mat3 = np.array(list).reshape(3,3)
    
  except:
    raise ValueError("List must contain nine numbers.")

  calculations = {}
  for key, operation in {
    'mean': 'mean',
    'variance': 'var',
    'standard deviation': 'std',
    'max': 'max',
    'min': 'min', 
    'sum': 'sum'
  }.items():
    calculations[key] = [
      getattr(mat3, operation)(axis=0).tolist(),
      getattr(mat3, operation)(axis=1).tolist(),
      getattr(mat3, operation)()
    ]

  return calculations