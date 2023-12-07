"""
vektor/core/dtype.py
"""
from typing import List, Tuple, Union

TBool = bool
TInt = int
TFloat = float
TComplex = complex
TScalar = Union[TBool, TInt, TFloat, TComplex]
TVector = List[TScalar]
TMatrix = List[TVector]
TTensor2D = TMatrix  # 2D tensor (matrix)
TTensor3D = List[TMatrix]  # 3D tensor (list of matrices)
TShape = Tuple[TInt, TInt]
TNoise = Union[TInt, TFloat]
