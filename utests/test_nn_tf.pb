
B
input_0Placeholder*
dtype0*
shape:��
B
input_1Placeholder*
dtype0*
shape:��
%
subSubinput_1input_0*
T0

L2LossL2Losssub*
T0
6
	truediv/yConst*
dtype0*
valueB
 *  �?
.
truedivRealDivL2Loss	truediv/y*
T0
8
truediv_1/yConst*
dtype0*
valueB
 *   H
3
	truediv_1RealDivtruedivtruediv_1/y*
T0
8
truediv_2/yConst*
dtype0*
valueB
 *  �?
5
	truediv_2RealDiv	truediv_1truediv_2/y*
T0
7

zeros_likeConst*
dtype0*
valueB
 *    
K
stackPack	truediv_2
zeros_like*
N*
T0*
axis���������
$
output_0Identitystack*
T0
6
	grad_ys_0Placeholder*
dtype0*
shape:
3
gradients/grad_ys_0Identity	grad_ys_0*
T0
d
gradients/stack_grad/unstackUnpackgradients/grad_ys_0*
T0*
axis���������*	
num
G
gradients/truediv_2_grad/ShapeConst*
dtype0*
valueB 
I
 gradients/truediv_2_grad/Shape_1Const*
dtype0*
valueB 
�
.gradients/truediv_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/truediv_2_grad/Shape gradients/truediv_2_grad/Shape_1*
T0
_
 gradients/truediv_2_grad/RealDivRealDivgradients/stack_grad/unstacktruediv_2/y*
T0
�
gradients/truediv_2_grad/SumSum gradients/truediv_2_grad/RealDiv.gradients/truediv_2_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
 gradients/truediv_2_grad/ReshapeReshapegradients/truediv_2_grad/Sumgradients/truediv_2_grad/Shape*
T0*
Tshape0
7
gradients/truediv_2_grad/NegNeg	truediv_1*
T0
a
"gradients/truediv_2_grad/RealDiv_1RealDivgradients/truediv_2_grad/Negtruediv_2/y*
T0
g
"gradients/truediv_2_grad/RealDiv_2RealDiv"gradients/truediv_2_grad/RealDiv_1truediv_2/y*
T0
n
gradients/truediv_2_grad/mulMulgradients/stack_grad/unstack"gradients/truediv_2_grad/RealDiv_2*
T0
�
gradients/truediv_2_grad/Sum_1Sumgradients/truediv_2_grad/mul0gradients/truediv_2_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
"gradients/truediv_2_grad/Reshape_1Reshapegradients/truediv_2_grad/Sum_1 gradients/truediv_2_grad/Shape_1*
T0*
Tshape0
G
gradients/truediv_1_grad/ShapeConst*
dtype0*
valueB 
I
 gradients/truediv_1_grad/Shape_1Const*
dtype0*
valueB 
�
.gradients/truediv_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/truediv_1_grad/Shape gradients/truediv_1_grad/Shape_1*
T0
c
 gradients/truediv_1_grad/RealDivRealDiv gradients/truediv_2_grad/Reshapetruediv_1/y*
T0
�
gradients/truediv_1_grad/SumSum gradients/truediv_1_grad/RealDiv.gradients/truediv_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
 gradients/truediv_1_grad/ReshapeReshapegradients/truediv_1_grad/Sumgradients/truediv_1_grad/Shape*
T0*
Tshape0
5
gradients/truediv_1_grad/NegNegtruediv*
T0
a
"gradients/truediv_1_grad/RealDiv_1RealDivgradients/truediv_1_grad/Negtruediv_1/y*
T0
g
"gradients/truediv_1_grad/RealDiv_2RealDiv"gradients/truediv_1_grad/RealDiv_1truediv_1/y*
T0
r
gradients/truediv_1_grad/mulMul gradients/truediv_2_grad/Reshape"gradients/truediv_1_grad/RealDiv_2*
T0
�
gradients/truediv_1_grad/Sum_1Sumgradients/truediv_1_grad/mul0gradients/truediv_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
"gradients/truediv_1_grad/Reshape_1Reshapegradients/truediv_1_grad/Sum_1 gradients/truediv_1_grad/Shape_1*
T0*
Tshape0
E
gradients/truediv_grad/ShapeConst*
dtype0*
valueB 
G
gradients/truediv_grad/Shape_1Const*
dtype0*
valueB 
�
,gradients/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/truediv_grad/Shapegradients/truediv_grad/Shape_1*
T0
_
gradients/truediv_grad/RealDivRealDiv gradients/truediv_1_grad/Reshape	truediv/y*
T0
�
gradients/truediv_grad/SumSumgradients/truediv_grad/RealDiv,gradients/truediv_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
z
gradients/truediv_grad/ReshapeReshapegradients/truediv_grad/Sumgradients/truediv_grad/Shape*
T0*
Tshape0
2
gradients/truediv_grad/NegNegL2Loss*
T0
[
 gradients/truediv_grad/RealDiv_1RealDivgradients/truediv_grad/Neg	truediv/y*
T0
a
 gradients/truediv_grad/RealDiv_2RealDiv gradients/truediv_grad/RealDiv_1	truediv/y*
T0
n
gradients/truediv_grad/mulMul gradients/truediv_1_grad/Reshape gradients/truediv_grad/RealDiv_2*
T0
�
gradients/truediv_grad/Sum_1Sumgradients/truediv_grad/mul.gradients/truediv_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
 gradients/truediv_grad/Reshape_1Reshapegradients/truediv_grad/Sum_1gradients/truediv_grad/Shape_1*
T0*
Tshape0
N
gradients/L2Loss_grad/mulMulsubgradients/truediv_grad/Reshape*
T0
A
gradients/sub_grad/NegNeggradients/L2Loss_grad/mul*
T0
J
grad_0/inputPackgradients/sub_grad/Neg*
N*
T0*

axis 
<
grad_0Squeezegrad_0/input*
T0*
squeeze_dims
 
5
gradients_1/grad_ys_0Identity	grad_ys_0*
T0
h
gradients_1/stack_grad/unstackUnpackgradients_1/grad_ys_0*
T0*
axis���������*	
num
I
 gradients_1/truediv_2_grad/ShapeConst*
dtype0*
valueB 
K
"gradients_1/truediv_2_grad/Shape_1Const*
dtype0*
valueB 
�
0gradients_1/truediv_2_grad/BroadcastGradientArgsBroadcastGradientArgs gradients_1/truediv_2_grad/Shape"gradients_1/truediv_2_grad/Shape_1*
T0
c
"gradients_1/truediv_2_grad/RealDivRealDivgradients_1/stack_grad/unstacktruediv_2/y*
T0
�
gradients_1/truediv_2_grad/SumSum"gradients_1/truediv_2_grad/RealDiv0gradients_1/truediv_2_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
"gradients_1/truediv_2_grad/ReshapeReshapegradients_1/truediv_2_grad/Sum gradients_1/truediv_2_grad/Shape*
T0*
Tshape0
9
gradients_1/truediv_2_grad/NegNeg	truediv_1*
T0
e
$gradients_1/truediv_2_grad/RealDiv_1RealDivgradients_1/truediv_2_grad/Negtruediv_2/y*
T0
k
$gradients_1/truediv_2_grad/RealDiv_2RealDiv$gradients_1/truediv_2_grad/RealDiv_1truediv_2/y*
T0
t
gradients_1/truediv_2_grad/mulMulgradients_1/stack_grad/unstack$gradients_1/truediv_2_grad/RealDiv_2*
T0
�
 gradients_1/truediv_2_grad/Sum_1Sumgradients_1/truediv_2_grad/mul2gradients_1/truediv_2_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
$gradients_1/truediv_2_grad/Reshape_1Reshape gradients_1/truediv_2_grad/Sum_1"gradients_1/truediv_2_grad/Shape_1*
T0*
Tshape0
I
 gradients_1/truediv_1_grad/ShapeConst*
dtype0*
valueB 
K
"gradients_1/truediv_1_grad/Shape_1Const*
dtype0*
valueB 
�
0gradients_1/truediv_1_grad/BroadcastGradientArgsBroadcastGradientArgs gradients_1/truediv_1_grad/Shape"gradients_1/truediv_1_grad/Shape_1*
T0
g
"gradients_1/truediv_1_grad/RealDivRealDiv"gradients_1/truediv_2_grad/Reshapetruediv_1/y*
T0
�
gradients_1/truediv_1_grad/SumSum"gradients_1/truediv_1_grad/RealDiv0gradients_1/truediv_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
"gradients_1/truediv_1_grad/ReshapeReshapegradients_1/truediv_1_grad/Sum gradients_1/truediv_1_grad/Shape*
T0*
Tshape0
7
gradients_1/truediv_1_grad/NegNegtruediv*
T0
e
$gradients_1/truediv_1_grad/RealDiv_1RealDivgradients_1/truediv_1_grad/Negtruediv_1/y*
T0
k
$gradients_1/truediv_1_grad/RealDiv_2RealDiv$gradients_1/truediv_1_grad/RealDiv_1truediv_1/y*
T0
x
gradients_1/truediv_1_grad/mulMul"gradients_1/truediv_2_grad/Reshape$gradients_1/truediv_1_grad/RealDiv_2*
T0
�
 gradients_1/truediv_1_grad/Sum_1Sumgradients_1/truediv_1_grad/mul2gradients_1/truediv_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
$gradients_1/truediv_1_grad/Reshape_1Reshape gradients_1/truediv_1_grad/Sum_1"gradients_1/truediv_1_grad/Shape_1*
T0*
Tshape0
G
gradients_1/truediv_grad/ShapeConst*
dtype0*
valueB 
I
 gradients_1/truediv_grad/Shape_1Const*
dtype0*
valueB 
�
.gradients_1/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/truediv_grad/Shape gradients_1/truediv_grad/Shape_1*
T0
c
 gradients_1/truediv_grad/RealDivRealDiv"gradients_1/truediv_1_grad/Reshape	truediv/y*
T0
�
gradients_1/truediv_grad/SumSum gradients_1/truediv_grad/RealDiv.gradients_1/truediv_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
 gradients_1/truediv_grad/ReshapeReshapegradients_1/truediv_grad/Sumgradients_1/truediv_grad/Shape*
T0*
Tshape0
4
gradients_1/truediv_grad/NegNegL2Loss*
T0
_
"gradients_1/truediv_grad/RealDiv_1RealDivgradients_1/truediv_grad/Neg	truediv/y*
T0
e
"gradients_1/truediv_grad/RealDiv_2RealDiv"gradients_1/truediv_grad/RealDiv_1	truediv/y*
T0
t
gradients_1/truediv_grad/mulMul"gradients_1/truediv_1_grad/Reshape"gradients_1/truediv_grad/RealDiv_2*
T0
�
gradients_1/truediv_grad/Sum_1Sumgradients_1/truediv_grad/mul0gradients_1/truediv_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
"gradients_1/truediv_grad/Reshape_1Reshapegradients_1/truediv_grad/Sum_1 gradients_1/truediv_grad/Shape_1*
T0*
Tshape0
R
gradients_1/L2Loss_grad/mulMulsub gradients_1/truediv_grad/Reshape*
T0
E
gradients_1/sub_grad/NegNeggradients_1/L2Loss_grad/mul*
T0
O
grad_1/inputPackgradients_1/L2Loss_grad/mul*
N*
T0*

axis 
<
grad_1Squeezegrad_1/input*
T0*
squeeze_dims
 

initNoOp"�