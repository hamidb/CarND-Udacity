       �K"	  �To�Abrain.Event:2��u��      B��	��To�A"��
e
PlaceholderPlaceholder*
dtype0*
shape: */
_output_shapes
:���������  
[
Placeholder_1Placeholder*
dtype0*
shape: *#
_output_shapes
:���������
�
wc1Variable*
shape: *
shared_name *
dtype0*
	container *&
_output_shapes
: 
�
&wc1/Initializer/truncated_normal/shapeConst*
dtype0*
_class

loc:@wc1*
_output_shapes
:*%
valueB"             
�
%wc1/Initializer/truncated_normal/meanConst*
dtype0*
_class

loc:@wc1*
_output_shapes
: *
valueB
 *    
�
'wc1/Initializer/truncated_normal/stddevConst*
dtype0*
_class

loc:@wc1*
_output_shapes
: *
valueB
 *
�#<
�
0wc1/Initializer/truncated_normal/TruncatedNormalTruncatedNormal&wc1/Initializer/truncated_normal/shape*
seed2 *
T0*
_class

loc:@wc1*
dtype0*

seed *&
_output_shapes
: 
�
$wc1/Initializer/truncated_normal/mulMul0wc1/Initializer/truncated_normal/TruncatedNormal'wc1/Initializer/truncated_normal/stddev*
T0*
_class

loc:@wc1*&
_output_shapes
: 
�
 wc1/Initializer/truncated_normalAdd$wc1/Initializer/truncated_normal/mul%wc1/Initializer/truncated_normal/mean*
T0*
_class

loc:@wc1*&
_output_shapes
: 
�

wc1/AssignAssignwc1 wc1/Initializer/truncated_normal*
use_locking(*
T0*
_class

loc:@wc1*&
_output_shapes
: *
validate_shape(
b
wc1/readIdentitywc1*
T0*
_class

loc:@wc1*&
_output_shapes
: 
w
wd1Variable*
shape:	�8@*
shared_name *
dtype0*
	container *
_output_shapes
:	�8@
�
&wd1/Initializer/truncated_normal/shapeConst*
dtype0*
_class

loc:@wd1*
_output_shapes
:*
valueB"   @   
�
%wd1/Initializer/truncated_normal/meanConst*
dtype0*
_class

loc:@wd1*
_output_shapes
: *
valueB
 *    
�
'wd1/Initializer/truncated_normal/stddevConst*
dtype0*
_class

loc:@wd1*
_output_shapes
: *
valueB
 *
�#<
�
0wd1/Initializer/truncated_normal/TruncatedNormalTruncatedNormal&wd1/Initializer/truncated_normal/shape*
seed2 *
T0*
_class

loc:@wd1*
dtype0*

seed *
_output_shapes
:	�8@
�
$wd1/Initializer/truncated_normal/mulMul0wd1/Initializer/truncated_normal/TruncatedNormal'wd1/Initializer/truncated_normal/stddev*
T0*
_class

loc:@wd1*
_output_shapes
:	�8@
�
 wd1/Initializer/truncated_normalAdd$wd1/Initializer/truncated_normal/mul%wd1/Initializer/truncated_normal/mean*
T0*
_class

loc:@wd1*
_output_shapes
:	�8@
�

wd1/AssignAssignwd1 wd1/Initializer/truncated_normal*
use_locking(*
T0*
_class

loc:@wd1*
_output_shapes
:	�8@*
validate_shape(
[
wd1/readIdentitywd1*
T0*
_class

loc:@wd1*
_output_shapes
:	�8@
t
woVariable*
shape
:@+*
shared_name *
dtype0*
	container *
_output_shapes

:@+
�
%wo/Initializer/truncated_normal/shapeConst*
dtype0*
_class
	loc:@wo*
_output_shapes
:*
valueB"@   +   
�
$wo/Initializer/truncated_normal/meanConst*
dtype0*
_class
	loc:@wo*
_output_shapes
: *
valueB
 *    
�
&wo/Initializer/truncated_normal/stddevConst*
dtype0*
_class
	loc:@wo*
_output_shapes
: *
valueB
 *
�#<
�
/wo/Initializer/truncated_normal/TruncatedNormalTruncatedNormal%wo/Initializer/truncated_normal/shape*
seed2 *
T0*
_class
	loc:@wo*
dtype0*

seed *
_output_shapes

:@+
�
#wo/Initializer/truncated_normal/mulMul/wo/Initializer/truncated_normal/TruncatedNormal&wo/Initializer/truncated_normal/stddev*
T0*
_class
	loc:@wo*
_output_shapes

:@+
�
wo/Initializer/truncated_normalAdd#wo/Initializer/truncated_normal/mul$wo/Initializer/truncated_normal/mean*
T0*
_class
	loc:@wo*
_output_shapes

:@+
�
	wo/AssignAssignwowo/Initializer/truncated_normal*
use_locking(*
T0*
_class
	loc:@wo*
_output_shapes

:@+*
validate_shape(
W
wo/readIdentitywo*
T0*
_class
	loc:@wo*
_output_shapes

:@+
m
bc1Variable*
shape: *
shared_name *
dtype0*
	container *
_output_shapes
: 
z
bc1/Initializer/ConstConst*
dtype0*
_class

loc:@bc1*
_output_shapes
: *
valueB *    
�

bc1/AssignAssignbc1bc1/Initializer/Const*
use_locking(*
T0*
_class

loc:@bc1*
_output_shapes
: *
validate_shape(
V
bc1/readIdentitybc1*
T0*
_class

loc:@bc1*
_output_shapes
: 
m
bd1Variable*
shape:@*
shared_name *
dtype0*
	container *
_output_shapes
:@
z
bd1/Initializer/ConstConst*
dtype0*
_class

loc:@bd1*
_output_shapes
:@*
valueB@*    
�

bd1/AssignAssignbd1bd1/Initializer/Const*
use_locking(*
T0*
_class

loc:@bd1*
_output_shapes
:@*
validate_shape(
V
bd1/readIdentitybd1*
T0*
_class

loc:@bd1*
_output_shapes
:@
l
boVariable*
shape:+*
shared_name *
dtype0*
	container *
_output_shapes
:+
x
bo/Initializer/ConstConst*
dtype0*
_class
	loc:@bo*
_output_shapes
:+*
valueB+*    
�
	bo/AssignAssignbobo/Initializer/Const*
use_locking(*
T0*
_class
	loc:@bo*
_output_shapes
:+*
validate_shape(
S
bo/readIdentitybo*
T0*
_class
	loc:@bo*
_output_shapes
:+
o
preprocess/rgb2gray/IdentityIdentityPlaceholder*
T0*/
_output_shapes
:���������  
Z
preprocess/rgb2gray/RankConst*
dtype0*
_output_shapes
: *
value	B :
[
preprocess/rgb2gray/sub/yConst*
dtype0*
_output_shapes
: *
value	B :
t
preprocess/rgb2gray/subSubpreprocess/rgb2gray/Rankpreprocess/rgb2gray/sub/y*
T0*
_output_shapes
: 
d
"preprocess/rgb2gray/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
value	B : 
�
preprocess/rgb2gray/ExpandDims
ExpandDimspreprocess/rgb2gray/sub"preprocess/rgb2gray/ExpandDims/dim*
T0*
_output_shapes
:*

Tdim0
n
preprocess/rgb2gray/mul/yConst*
dtype0*
_output_shapes
:*!
valueB"l	�>�E?�x�=
�
preprocess/rgb2gray/mulMulpreprocess/rgb2gray/Identitypreprocess/rgb2gray/mul/y*
T0*/
_output_shapes
:���������  
�
preprocess/rgb2gray/SumSumpreprocess/rgb2gray/mulpreprocess/rgb2gray/ExpandDims*
T0*
	keep_dims(*

Tidx0*/
_output_shapes
:���������  
r
preprocess/rgb2grayIdentitypreprocess/rgb2gray/Sum*
T0*/
_output_shapes
:���������  
s
preprocess/normalize/ConstConst*
dtype0*
_output_shapes
:*%
valueB"             
�
preprocess/normalize/MinMinpreprocess/rgb2graypreprocess/normalize/Const*
T0*
	keep_dims( *

Tidx0*
_output_shapes
: 
u
preprocess/normalize/Const_1Const*
dtype0*
_output_shapes
:*%
valueB"             
�
preprocess/normalize/MaxMaxpreprocess/rgb2graypreprocess/normalize/Const_1*
T0*
	keep_dims( *

Tidx0*
_output_shapes
: 
�
preprocess/normalize/SubSubpreprocess/rgb2graypreprocess/normalize/Min*
T0*/
_output_shapes
:���������  
v
preprocess/normalize/Sub_1Subpreprocess/normalize/Maxpreprocess/normalize/Min*
T0*
_output_shapes
: 
�
preprocess/normalize/DivDivpreprocess/normalize/Subpreprocess/normalize/Sub_1*
T0*/
_output_shapes
:���������  
�
conv1/Conv2DConv2Dpreprocess/normalize/Divwc1/read*
strides
*
paddingVALID*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*/
_output_shapes
:��������� 
b
	conv1/addAddconv1/Conv2Dbc1/read*
T0*/
_output_shapes
:��������� 
W

conv1/ReluRelu	conv1/add*
T0*/
_output_shapes
:��������� 
�
pooling/MaxPoolMaxPool
conv1/Relu*
strides
*
paddingVALID*
T0*/
_output_shapes
:��������� *
data_formatNHWC*
ksize

n
fully_connected/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"����   
�
fully_connected/ReshapeReshapepooling/MaxPoolfully_connected/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:����������8
�
fully_connected/MatMulMatMulfully_connected/Reshapewd1/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:���������@
n
fully_connected/AddAddfully_connected/MatMulbd1/read*
T0*'
_output_shapes
:���������@
c
fully_connected/ReluRelufully_connected/Add*
T0*'
_output_shapes
:���������@
V
dropout/keep_probConst*
dtype0*
_output_shapes
: *
valueB
 *  @?
a
dropout/ShapeShapefully_connected/Relu*
T0*
_output_shapes
:*
out_type0
_
dropout/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    
_
dropout/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape*

seed *
dtype0*
seed2 *0
_output_shapes
:������������������*
T0
z
dropout/random_uniform/subSubdropout/random_uniform/maxdropout/random_uniform/min*
T0*
_output_shapes
: 
�
dropout/random_uniform/mulMul$dropout/random_uniform/RandomUniformdropout/random_uniform/sub*
T0*0
_output_shapes
:������������������
�
dropout/random_uniformAdddropout/random_uniform/muldropout/random_uniform/min*
T0*0
_output_shapes
:������������������
x
dropout/addAdddropout/keep_probdropout/random_uniform*
T0*0
_output_shapes
:������������������
^
dropout/FloorFloordropout/add*
T0*0
_output_shapes
:������������������
m
dropout/DivDivfully_connected/Reludropout/keep_prob*
T0*'
_output_shapes
:���������@
`
dropout/mulMuldropout/Divdropout/Floor*
T0*'
_output_shapes
:���������@
�
output/MatMulMatMuldropout/mulwo/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:���������+
[

output/AddAddoutput/MatMulbo/read*
T0*'
_output_shapes
:���������+
e
loss/cross_entropy/ShapeShapePlaceholder_1*
T0*
_output_shapes
:*
out_type0
�
 loss/cross_entropy/cross_entropy#SparseSoftmaxCrossEntropyWithLogits
output/AddPlaceholder_1*
T0*
Tlabels0*6
_output_shapes$
":���������:���������+
T

loss/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
}
	loss/MeanMean loss/cross_entropy/cross_entropy
loss/Const*
T0*
	keep_dims( *

Tidx0*
_output_shapes
: 
\
loss/ScalarSummary/tagsConst*
dtype0*
_output_shapes
: *
valueB
 Bloss
h
loss/ScalarSummaryScalarSummaryloss/ScalarSummary/tags	loss/Mean*
T0*
_output_shapes
: 
X
predict/SoftmaxSoftmax
output/Add*
T0*'
_output_shapes
:���������+
Z
predict/ArgMax/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
}
predict/ArgMaxArgMaxpredict/Softmaxpredict/ArgMax/dimension*
T0*

Tidx0*#
_output_shapes
:���������
b
accuracy/CastCastpredict/ArgMax*

DstT0*

SrcT0	*#
_output_shapes
:���������
c
accuracy/EqualEqualaccuracy/CastPlaceholder_1*
T0*#
_output_shapes
:���������
d
accuracy/Cast_1Castaccuracy/Equal*

DstT0*

SrcT0
*#
_output_shapes
:���������
X
accuracy/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
t
accuracy/MeanMeanaccuracy/Cast_1accuracy/Const*
T0*
	keep_dims( *

Tidx0*
_output_shapes
: 
d
accuracy/ScalarSummary/tagsConst*
dtype0*
_output_shapes
: *
valueB Baccuracy
t
accuracy/ScalarSummaryScalarSummaryaccuracy/ScalarSummary/tagsaccuracy/Mean*
T0*
_output_shapes
: 
X
train/gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
Z
train/gradients/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
k
train/gradients/FillFilltrain/gradients/Shapetrain/gradients/Const*
T0*
_output_shapes
: 
v
,train/gradients/loss/Mean_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
�
&train/gradients/loss/Mean_grad/ReshapeReshapetrain/gradients/Fill,train/gradients/loss/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
�
$train/gradients/loss/Mean_grad/ShapeShape loss/cross_entropy/cross_entropy*
T0*
_output_shapes
:*
out_type0
�
#train/gradients/loss/Mean_grad/TileTile&train/gradients/loss/Mean_grad/Reshape$train/gradients/loss/Mean_grad/Shape*
T0*

Tmultiples0*#
_output_shapes
:���������
�
&train/gradients/loss/Mean_grad/Shape_1Shape loss/cross_entropy/cross_entropy*
T0*
_output_shapes
:*
out_type0
i
&train/gradients/loss/Mean_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
n
$train/gradients/loss/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
#train/gradients/loss/Mean_grad/ProdProd&train/gradients/loss/Mean_grad/Shape_1$train/gradients/loss/Mean_grad/Const*
T0*
	keep_dims( *

Tidx0*
_output_shapes
: 
p
&train/gradients/loss/Mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
%train/gradients/loss/Mean_grad/Prod_1Prod&train/gradients/loss/Mean_grad/Shape_2&train/gradients/loss/Mean_grad/Const_1*
T0*
	keep_dims( *

Tidx0*
_output_shapes
: 
j
(train/gradients/loss/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
�
&train/gradients/loss/Mean_grad/MaximumMaximum%train/gradients/loss/Mean_grad/Prod_1(train/gradients/loss/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
�
'train/gradients/loss/Mean_grad/floordivDiv#train/gradients/loss/Mean_grad/Prod&train/gradients/loss/Mean_grad/Maximum*
T0*
_output_shapes
: 
�
#train/gradients/loss/Mean_grad/CastCast'train/gradients/loss/Mean_grad/floordiv*

DstT0*

SrcT0*
_output_shapes
: 
�
&train/gradients/loss/Mean_grad/truedivDiv#train/gradients/loss/Mean_grad/Tile#train/gradients/loss/Mean_grad/Cast*
T0*#
_output_shapes
:���������
}
train/gradients/zeros_like	ZerosLike"loss/cross_entropy/cross_entropy:1*
T0*'
_output_shapes
:���������+
�
Dtrain/gradients/loss/cross_entropy/cross_entropy_grad/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
���������
�
@train/gradients/loss/cross_entropy/cross_entropy_grad/ExpandDims
ExpandDims&train/gradients/loss/Mean_grad/truedivDtrain/gradients/loss/cross_entropy/cross_entropy_grad/ExpandDims/dim*
T0*'
_output_shapes
:���������*

Tdim0
�
9train/gradients/loss/cross_entropy/cross_entropy_grad/mulMul@train/gradients/loss/cross_entropy/cross_entropy_grad/ExpandDims"loss/cross_entropy/cross_entropy:1*
T0*'
_output_shapes
:���������+
r
%train/gradients/output/Add_grad/ShapeShapeoutput/MatMul*
T0*
_output_shapes
:*
out_type0
q
'train/gradients/output/Add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:+
�
5train/gradients/output/Add_grad/BroadcastGradientArgsBroadcastGradientArgs%train/gradients/output/Add_grad/Shape'train/gradients/output/Add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
#train/gradients/output/Add_grad/SumSum9train/gradients/loss/cross_entropy/cross_entropy_grad/mul5train/gradients/output/Add_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0*
_output_shapes
:
�
'train/gradients/output/Add_grad/ReshapeReshape#train/gradients/output/Add_grad/Sum%train/gradients/output/Add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������+
�
%train/gradients/output/Add_grad/Sum_1Sum9train/gradients/loss/cross_entropy/cross_entropy_grad/mul7train/gradients/output/Add_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0*
_output_shapes
:
�
)train/gradients/output/Add_grad/Reshape_1Reshape%train/gradients/output/Add_grad/Sum_1'train/gradients/output/Add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:+
�
0train/gradients/output/Add_grad/tuple/group_depsNoOp(^train/gradients/output/Add_grad/Reshape*^train/gradients/output/Add_grad/Reshape_1
�
8train/gradients/output/Add_grad/tuple/control_dependencyIdentity'train/gradients/output/Add_grad/Reshape1^train/gradients/output/Add_grad/tuple/group_deps*
T0*:
_class0
.,loc:@train/gradients/output/Add_grad/Reshape*'
_output_shapes
:���������+
�
:train/gradients/output/Add_grad/tuple/control_dependency_1Identity)train/gradients/output/Add_grad/Reshape_11^train/gradients/output/Add_grad/tuple/group_deps*
T0*<
_class2
0.loc:@train/gradients/output/Add_grad/Reshape_1*
_output_shapes
:+
�
)train/gradients/output/MatMul_grad/MatMulMatMul8train/gradients/output/Add_grad/tuple/control_dependencywo/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:���������@
�
+train/gradients/output/MatMul_grad/MatMul_1MatMuldropout/mul8train/gradients/output/Add_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes

:@+
�
3train/gradients/output/MatMul_grad/tuple/group_depsNoOp*^train/gradients/output/MatMul_grad/MatMul,^train/gradients/output/MatMul_grad/MatMul_1
�
;train/gradients/output/MatMul_grad/tuple/control_dependencyIdentity)train/gradients/output/MatMul_grad/MatMul4^train/gradients/output/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@train/gradients/output/MatMul_grad/MatMul*'
_output_shapes
:���������@
�
=train/gradients/output/MatMul_grad/tuple/control_dependency_1Identity+train/gradients/output/MatMul_grad/MatMul_14^train/gradients/output/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@train/gradients/output/MatMul_grad/MatMul_1*
_output_shapes

:@+
q
&train/gradients/dropout/mul_grad/ShapeShapedropout/Div*
T0*
_output_shapes
:*
out_type0
u
(train/gradients/dropout/mul_grad/Shape_1Shapedropout/Floor*
T0*
_output_shapes
:*
out_type0
�
6train/gradients/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/dropout/mul_grad/Shape(train/gradients/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$train/gradients/dropout/mul_grad/mulMul;train/gradients/output/MatMul_grad/tuple/control_dependencydropout/Floor*
T0*'
_output_shapes
:���������@
�
$train/gradients/dropout/mul_grad/SumSum$train/gradients/dropout/mul_grad/mul6train/gradients/dropout/mul_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0*
_output_shapes
:
�
(train/gradients/dropout/mul_grad/ReshapeReshape$train/gradients/dropout/mul_grad/Sum&train/gradients/dropout/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������@
�
&train/gradients/dropout/mul_grad/mul_1Muldropout/Div;train/gradients/output/MatMul_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������@
�
&train/gradients/dropout/mul_grad/Sum_1Sum&train/gradients/dropout/mul_grad/mul_18train/gradients/dropout/mul_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0*
_output_shapes
:
�
*train/gradients/dropout/mul_grad/Reshape_1Reshape&train/gradients/dropout/mul_grad/Sum_1(train/gradients/dropout/mul_grad/Shape_1*
T0*
Tshape0*0
_output_shapes
:������������������
�
1train/gradients/dropout/mul_grad/tuple/group_depsNoOp)^train/gradients/dropout/mul_grad/Reshape+^train/gradients/dropout/mul_grad/Reshape_1
�
9train/gradients/dropout/mul_grad/tuple/control_dependencyIdentity(train/gradients/dropout/mul_grad/Reshape2^train/gradients/dropout/mul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/dropout/mul_grad/Reshape*'
_output_shapes
:���������@
�
;train/gradients/dropout/mul_grad/tuple/control_dependency_1Identity*train/gradients/dropout/mul_grad/Reshape_12^train/gradients/dropout/mul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/dropout/mul_grad/Reshape_1*0
_output_shapes
:������������������
z
&train/gradients/dropout/Div_grad/ShapeShapefully_connected/Relu*
T0*
_output_shapes
:*
out_type0
k
(train/gradients/dropout/Div_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
�
6train/gradients/dropout/Div_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/dropout/Div_grad/Shape(train/gradients/dropout/Div_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
(train/gradients/dropout/Div_grad/truedivDiv9train/gradients/dropout/mul_grad/tuple/control_dependencydropout/keep_prob*
T0*'
_output_shapes
:���������@
�
$train/gradients/dropout/Div_grad/SumSum(train/gradients/dropout/Div_grad/truediv6train/gradients/dropout/Div_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0*
_output_shapes
:
�
(train/gradients/dropout/Div_grad/ReshapeReshape$train/gradients/dropout/Div_grad/Sum&train/gradients/dropout/Div_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������@
s
$train/gradients/dropout/Div_grad/NegNegfully_connected/Relu*
T0*'
_output_shapes
:���������@
e
'train/gradients/dropout/Div_grad/SquareSquaredropout/keep_prob*
T0*
_output_shapes
: 
�
*train/gradients/dropout/Div_grad/truediv_1Div$train/gradients/dropout/Div_grad/Neg'train/gradients/dropout/Div_grad/Square*
T0*'
_output_shapes
:���������@
�
$train/gradients/dropout/Div_grad/mulMul9train/gradients/dropout/mul_grad/tuple/control_dependency*train/gradients/dropout/Div_grad/truediv_1*
T0*'
_output_shapes
:���������@
�
&train/gradients/dropout/Div_grad/Sum_1Sum$train/gradients/dropout/Div_grad/mul8train/gradients/dropout/Div_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0*
_output_shapes
:
�
*train/gradients/dropout/Div_grad/Reshape_1Reshape&train/gradients/dropout/Div_grad/Sum_1(train/gradients/dropout/Div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
1train/gradients/dropout/Div_grad/tuple/group_depsNoOp)^train/gradients/dropout/Div_grad/Reshape+^train/gradients/dropout/Div_grad/Reshape_1
�
9train/gradients/dropout/Div_grad/tuple/control_dependencyIdentity(train/gradients/dropout/Div_grad/Reshape2^train/gradients/dropout/Div_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/dropout/Div_grad/Reshape*'
_output_shapes
:���������@
�
;train/gradients/dropout/Div_grad/tuple/control_dependency_1Identity*train/gradients/dropout/Div_grad/Reshape_12^train/gradients/dropout/Div_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/dropout/Div_grad/Reshape_1*
_output_shapes
: 
�
2train/gradients/fully_connected/Relu_grad/ReluGradReluGrad9train/gradients/dropout/Div_grad/tuple/control_dependencyfully_connected/Relu*
T0*'
_output_shapes
:���������@
�
.train/gradients/fully_connected/Add_grad/ShapeShapefully_connected/MatMul*
T0*
_output_shapes
:*
out_type0
z
0train/gradients/fully_connected/Add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:@
�
>train/gradients/fully_connected/Add_grad/BroadcastGradientArgsBroadcastGradientArgs.train/gradients/fully_connected/Add_grad/Shape0train/gradients/fully_connected/Add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
,train/gradients/fully_connected/Add_grad/SumSum2train/gradients/fully_connected/Relu_grad/ReluGrad>train/gradients/fully_connected/Add_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0*
_output_shapes
:
�
0train/gradients/fully_connected/Add_grad/ReshapeReshape,train/gradients/fully_connected/Add_grad/Sum.train/gradients/fully_connected/Add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������@
�
.train/gradients/fully_connected/Add_grad/Sum_1Sum2train/gradients/fully_connected/Relu_grad/ReluGrad@train/gradients/fully_connected/Add_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0*
_output_shapes
:
�
2train/gradients/fully_connected/Add_grad/Reshape_1Reshape.train/gradients/fully_connected/Add_grad/Sum_10train/gradients/fully_connected/Add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:@
�
9train/gradients/fully_connected/Add_grad/tuple/group_depsNoOp1^train/gradients/fully_connected/Add_grad/Reshape3^train/gradients/fully_connected/Add_grad/Reshape_1
�
Atrain/gradients/fully_connected/Add_grad/tuple/control_dependencyIdentity0train/gradients/fully_connected/Add_grad/Reshape:^train/gradients/fully_connected/Add_grad/tuple/group_deps*
T0*C
_class9
75loc:@train/gradients/fully_connected/Add_grad/Reshape*'
_output_shapes
:���������@
�
Ctrain/gradients/fully_connected/Add_grad/tuple/control_dependency_1Identity2train/gradients/fully_connected/Add_grad/Reshape_1:^train/gradients/fully_connected/Add_grad/tuple/group_deps*
T0*E
_class;
97loc:@train/gradients/fully_connected/Add_grad/Reshape_1*
_output_shapes
:@
�
2train/gradients/fully_connected/MatMul_grad/MatMulMatMulAtrain/gradients/fully_connected/Add_grad/tuple/control_dependencywd1/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:����������8
�
4train/gradients/fully_connected/MatMul_grad/MatMul_1MatMulfully_connected/ReshapeAtrain/gradients/fully_connected/Add_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes
:	�8@
�
<train/gradients/fully_connected/MatMul_grad/tuple/group_depsNoOp3^train/gradients/fully_connected/MatMul_grad/MatMul5^train/gradients/fully_connected/MatMul_grad/MatMul_1
�
Dtrain/gradients/fully_connected/MatMul_grad/tuple/control_dependencyIdentity2train/gradients/fully_connected/MatMul_grad/MatMul=^train/gradients/fully_connected/MatMul_grad/tuple/group_deps*
T0*E
_class;
97loc:@train/gradients/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������8
�
Ftrain/gradients/fully_connected/MatMul_grad/tuple/control_dependency_1Identity4train/gradients/fully_connected/MatMul_grad/MatMul_1=^train/gradients/fully_connected/MatMul_grad/tuple/group_deps*
T0*G
_class=
;9loc:@train/gradients/fully_connected/MatMul_grad/MatMul_1*
_output_shapes
:	�8@
�
2train/gradients/fully_connected/Reshape_grad/ShapeShapepooling/MaxPool*
T0*
_output_shapes
:*
out_type0
�
4train/gradients/fully_connected/Reshape_grad/ReshapeReshapeDtrain/gradients/fully_connected/MatMul_grad/tuple/control_dependency2train/gradients/fully_connected/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:��������� 
�
0train/gradients/pooling/MaxPool_grad/MaxPoolGradMaxPoolGrad
conv1/Relupooling/MaxPool4train/gradients/fully_connected/Reshape_grad/Reshape*
strides
*
paddingVALID*
T0*/
_output_shapes
:��������� *
data_formatNHWC*
ksize

�
(train/gradients/conv1/Relu_grad/ReluGradReluGrad0train/gradients/pooling/MaxPool_grad/MaxPoolGrad
conv1/Relu*
T0*/
_output_shapes
:��������� 
p
$train/gradients/conv1/add_grad/ShapeShapeconv1/Conv2D*
T0*
_output_shapes
:*
out_type0
p
&train/gradients/conv1/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
4train/gradients/conv1/add_grad/BroadcastGradientArgsBroadcastGradientArgs$train/gradients/conv1/add_grad/Shape&train/gradients/conv1/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
"train/gradients/conv1/add_grad/SumSum(train/gradients/conv1/Relu_grad/ReluGrad4train/gradients/conv1/add_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0*
_output_shapes
:
�
&train/gradients/conv1/add_grad/ReshapeReshape"train/gradients/conv1/add_grad/Sum$train/gradients/conv1/add_grad/Shape*
T0*
Tshape0*/
_output_shapes
:��������� 
�
$train/gradients/conv1/add_grad/Sum_1Sum(train/gradients/conv1/Relu_grad/ReluGrad6train/gradients/conv1/add_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0*
_output_shapes
:
�
(train/gradients/conv1/add_grad/Reshape_1Reshape$train/gradients/conv1/add_grad/Sum_1&train/gradients/conv1/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
/train/gradients/conv1/add_grad/tuple/group_depsNoOp'^train/gradients/conv1/add_grad/Reshape)^train/gradients/conv1/add_grad/Reshape_1
�
7train/gradients/conv1/add_grad/tuple/control_dependencyIdentity&train/gradients/conv1/add_grad/Reshape0^train/gradients/conv1/add_grad/tuple/group_deps*
T0*9
_class/
-+loc:@train/gradients/conv1/add_grad/Reshape*/
_output_shapes
:��������� 
�
9train/gradients/conv1/add_grad/tuple/control_dependency_1Identity(train/gradients/conv1/add_grad/Reshape_10^train/gradients/conv1/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/conv1/add_grad/Reshape_1*
_output_shapes
: 

'train/gradients/conv1/Conv2D_grad/ShapeShapepreprocess/normalize/Div*
T0*
_output_shapes
:*
out_type0
�
5train/gradients/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput'train/gradients/conv1/Conv2D_grad/Shapewc1/read7train/gradients/conv1/add_grad/tuple/control_dependency*
strides
*
paddingVALID*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*J
_output_shapes8
6:4������������������������������������
�
)train/gradients/conv1/Conv2D_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"             
�
6train/gradients/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterpreprocess/normalize/Div)train/gradients/conv1/Conv2D_grad/Shape_17train/gradients/conv1/add_grad/tuple/control_dependency*
strides
*
paddingVALID*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*&
_output_shapes
: 
�
2train/gradients/conv1/Conv2D_grad/tuple/group_depsNoOp6^train/gradients/conv1/Conv2D_grad/Conv2DBackpropInput7^train/gradients/conv1/Conv2D_grad/Conv2DBackpropFilter
�
:train/gradients/conv1/Conv2D_grad/tuple/control_dependencyIdentity5train/gradients/conv1/Conv2D_grad/Conv2DBackpropInput3^train/gradients/conv1/Conv2D_grad/tuple/group_deps*
T0*H
_class>
<:loc:@train/gradients/conv1/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������  
�
<train/gradients/conv1/Conv2D_grad/tuple/control_dependency_1Identity6train/gradients/conv1/Conv2D_grad/Conv2DBackpropFilter3^train/gradients/conv1/Conv2D_grad/tuple/group_deps*
T0*I
_class?
=;loc:@train/gradients/conv1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: 
|
train/beta1_power/initial_valueConst*
dtype0*
_class

loc:@wc1*
_output_shapes
: *
valueB
 *fff?
�
train/beta1_powerVariable*
shape: *
_class

loc:@wc1*
	container *
shared_name *
dtype0*
_output_shapes
: 
�
train/beta1_power/AssignAssigntrain/beta1_powertrain/beta1_power/initial_value*
use_locking(*
T0*
_class

loc:@wc1*
_output_shapes
: *
validate_shape(
n
train/beta1_power/readIdentitytrain/beta1_power*
T0*
_class

loc:@wc1*
_output_shapes
: 
|
train/beta2_power/initial_valueConst*
dtype0*
_class

loc:@wc1*
_output_shapes
: *
valueB
 *w�?
�
train/beta2_powerVariable*
shape: *
_class

loc:@wc1*
	container *
shared_name *
dtype0*
_output_shapes
: 
�
train/beta2_power/AssignAssigntrain/beta2_powertrain/beta2_power/initial_value*
use_locking(*
T0*
_class

loc:@wc1*
_output_shapes
: *
validate_shape(
n
train/beta2_power/readIdentitytrain/beta2_power*
T0*
_class

loc:@wc1*
_output_shapes
: 
p
train/zerosConst*
dtype0*&
_output_shapes
: *%
valueB *    
�
train/wc1/AdamVariable*
shape: *
_class

loc:@wc1*
	container *
shared_name *
dtype0*&
_output_shapes
: 
�
train/wc1/Adam/AssignAssigntrain/wc1/Adamtrain/zeros*
use_locking(*
T0*
_class

loc:@wc1*&
_output_shapes
: *
validate_shape(
x
train/wc1/Adam/readIdentitytrain/wc1/Adam*
T0*
_class

loc:@wc1*&
_output_shapes
: 
r
train/zeros_1Const*
dtype0*&
_output_shapes
: *%
valueB *    
�
train/wc1/Adam_1Variable*
shape: *
_class

loc:@wc1*
	container *
shared_name *
dtype0*&
_output_shapes
: 
�
train/wc1/Adam_1/AssignAssigntrain/wc1/Adam_1train/zeros_1*
use_locking(*
T0*
_class

loc:@wc1*&
_output_shapes
: *
validate_shape(
|
train/wc1/Adam_1/readIdentitytrain/wc1/Adam_1*
T0*
_class

loc:@wc1*&
_output_shapes
: 
d
train/zeros_2Const*
dtype0*
_output_shapes
:	�8@*
valueB	�8@*    
�
train/wd1/AdamVariable*
shape:	�8@*
_class

loc:@wd1*
	container *
shared_name *
dtype0*
_output_shapes
:	�8@
�
train/wd1/Adam/AssignAssigntrain/wd1/Adamtrain/zeros_2*
use_locking(*
T0*
_class

loc:@wd1*
_output_shapes
:	�8@*
validate_shape(
q
train/wd1/Adam/readIdentitytrain/wd1/Adam*
T0*
_class

loc:@wd1*
_output_shapes
:	�8@
d
train/zeros_3Const*
dtype0*
_output_shapes
:	�8@*
valueB	�8@*    
�
train/wd1/Adam_1Variable*
shape:	�8@*
_class

loc:@wd1*
	container *
shared_name *
dtype0*
_output_shapes
:	�8@
�
train/wd1/Adam_1/AssignAssigntrain/wd1/Adam_1train/zeros_3*
use_locking(*
T0*
_class

loc:@wd1*
_output_shapes
:	�8@*
validate_shape(
u
train/wd1/Adam_1/readIdentitytrain/wd1/Adam_1*
T0*
_class

loc:@wd1*
_output_shapes
:	�8@
b
train/zeros_4Const*
dtype0*
_output_shapes

:@+*
valueB@+*    
�
train/wo/AdamVariable*
shape
:@+*
_class
	loc:@wo*
	container *
shared_name *
dtype0*
_output_shapes

:@+
�
train/wo/Adam/AssignAssigntrain/wo/Adamtrain/zeros_4*
use_locking(*
T0*
_class
	loc:@wo*
_output_shapes

:@+*
validate_shape(
m
train/wo/Adam/readIdentitytrain/wo/Adam*
T0*
_class
	loc:@wo*
_output_shapes

:@+
b
train/zeros_5Const*
dtype0*
_output_shapes

:@+*
valueB@+*    
�
train/wo/Adam_1Variable*
shape
:@+*
_class
	loc:@wo*
	container *
shared_name *
dtype0*
_output_shapes

:@+
�
train/wo/Adam_1/AssignAssigntrain/wo/Adam_1train/zeros_5*
use_locking(*
T0*
_class
	loc:@wo*
_output_shapes

:@+*
validate_shape(
q
train/wo/Adam_1/readIdentitytrain/wo/Adam_1*
T0*
_class
	loc:@wo*
_output_shapes

:@+
Z
train/zeros_6Const*
dtype0*
_output_shapes
: *
valueB *    
�
train/bc1/AdamVariable*
shape: *
_class

loc:@bc1*
	container *
shared_name *
dtype0*
_output_shapes
: 
�
train/bc1/Adam/AssignAssigntrain/bc1/Adamtrain/zeros_6*
use_locking(*
T0*
_class

loc:@bc1*
_output_shapes
: *
validate_shape(
l
train/bc1/Adam/readIdentitytrain/bc1/Adam*
T0*
_class

loc:@bc1*
_output_shapes
: 
Z
train/zeros_7Const*
dtype0*
_output_shapes
: *
valueB *    
�
train/bc1/Adam_1Variable*
shape: *
_class

loc:@bc1*
	container *
shared_name *
dtype0*
_output_shapes
: 
�
train/bc1/Adam_1/AssignAssigntrain/bc1/Adam_1train/zeros_7*
use_locking(*
T0*
_class

loc:@bc1*
_output_shapes
: *
validate_shape(
p
train/bc1/Adam_1/readIdentitytrain/bc1/Adam_1*
T0*
_class

loc:@bc1*
_output_shapes
: 
Z
train/zeros_8Const*
dtype0*
_output_shapes
:@*
valueB@*    
�
train/bd1/AdamVariable*
shape:@*
_class

loc:@bd1*
	container *
shared_name *
dtype0*
_output_shapes
:@
�
train/bd1/Adam/AssignAssigntrain/bd1/Adamtrain/zeros_8*
use_locking(*
T0*
_class

loc:@bd1*
_output_shapes
:@*
validate_shape(
l
train/bd1/Adam/readIdentitytrain/bd1/Adam*
T0*
_class

loc:@bd1*
_output_shapes
:@
Z
train/zeros_9Const*
dtype0*
_output_shapes
:@*
valueB@*    
�
train/bd1/Adam_1Variable*
shape:@*
_class

loc:@bd1*
	container *
shared_name *
dtype0*
_output_shapes
:@
�
train/bd1/Adam_1/AssignAssigntrain/bd1/Adam_1train/zeros_9*
use_locking(*
T0*
_class

loc:@bd1*
_output_shapes
:@*
validate_shape(
p
train/bd1/Adam_1/readIdentitytrain/bd1/Adam_1*
T0*
_class

loc:@bd1*
_output_shapes
:@
[
train/zeros_10Const*
dtype0*
_output_shapes
:+*
valueB+*    
�
train/bo/AdamVariable*
shape:+*
_class
	loc:@bo*
	container *
shared_name *
dtype0*
_output_shapes
:+
�
train/bo/Adam/AssignAssigntrain/bo/Adamtrain/zeros_10*
use_locking(*
T0*
_class
	loc:@bo*
_output_shapes
:+*
validate_shape(
i
train/bo/Adam/readIdentitytrain/bo/Adam*
T0*
_class
	loc:@bo*
_output_shapes
:+
[
train/zeros_11Const*
dtype0*
_output_shapes
:+*
valueB+*    
�
train/bo/Adam_1Variable*
shape:+*
_class
	loc:@bo*
	container *
shared_name *
dtype0*
_output_shapes
:+
�
train/bo/Adam_1/AssignAssigntrain/bo/Adam_1train/zeros_11*
use_locking(*
T0*
_class
	loc:@bo*
_output_shapes
:+*
validate_shape(
m
train/bo/Adam_1/readIdentitytrain/bo/Adam_1*
T0*
_class
	loc:@bo*
_output_shapes
:+
]
train/Adam/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *o�:
U
train/Adam/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
U
train/Adam/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *w�?
W
train/Adam/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *w�+2
�
train/Adam/update_wc1/ApplyAdam	ApplyAdamwc1train/wc1/Adamtrain/wc1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon<train/gradients/conv1/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class

loc:@wc1*&
_output_shapes
: 
�
train/Adam/update_wd1/ApplyAdam	ApplyAdamwd1train/wd1/Adamtrain/wd1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonFtrain/gradients/fully_connected/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class

loc:@wd1*
_output_shapes
:	�8@
�
train/Adam/update_wo/ApplyAdam	ApplyAdamwotrain/wo/Adamtrain/wo/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon=train/gradients/output/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
	loc:@wo*
_output_shapes

:@+
�
train/Adam/update_bc1/ApplyAdam	ApplyAdambc1train/bc1/Adamtrain/bc1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon9train/gradients/conv1/add_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class

loc:@bc1*
_output_shapes
: 
�
train/Adam/update_bd1/ApplyAdam	ApplyAdambd1train/bd1/Adamtrain/bd1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonCtrain/gradients/fully_connected/Add_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class

loc:@bd1*
_output_shapes
:@
�
train/Adam/update_bo/ApplyAdam	ApplyAdambotrain/bo/Adamtrain/bo/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon:train/gradients/output/Add_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
	loc:@bo*
_output_shapes
:+
�
train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta1 ^train/Adam/update_wc1/ApplyAdam ^train/Adam/update_wd1/ApplyAdam^train/Adam/update_wo/ApplyAdam ^train/Adam/update_bc1/ApplyAdam ^train/Adam/update_bd1/ApplyAdam^train/Adam/update_bo/ApplyAdam*
T0*
_class

loc:@wc1*
_output_shapes
: 
�
train/Adam/AssignAssigntrain/beta1_powertrain/Adam/mul*
use_locking( *
T0*
_class

loc:@wc1*
_output_shapes
: *
validate_shape(
�
train/Adam/mul_1Multrain/beta2_power/readtrain/Adam/beta2 ^train/Adam/update_wc1/ApplyAdam ^train/Adam/update_wd1/ApplyAdam^train/Adam/update_wo/ApplyAdam ^train/Adam/update_bc1/ApplyAdam ^train/Adam/update_bd1/ApplyAdam^train/Adam/update_bo/ApplyAdam*
T0*
_class

loc:@wc1*
_output_shapes
: 
�
train/Adam/Assign_1Assigntrain/beta2_powertrain/Adam/mul_1*
use_locking( *
T0*
_class

loc:@wc1*
_output_shapes
: *
validate_shape(
�

train/AdamNoOp ^train/Adam/update_wc1/ApplyAdam ^train/Adam/update_wd1/ApplyAdam^train/Adam/update_wo/ApplyAdam ^train/Adam/update_bc1/ApplyAdam ^train/Adam/update_bd1/ApplyAdam^train/Adam/update_bo/ApplyAdam^train/Adam/Assign^train/Adam/Assign_1
�
initializer/initNoOp^wc1/Assign^wd1/Assign
^wo/Assign^bc1/Assign^bd1/Assign
^bo/Assign^train/beta1_power/Assign^train/beta2_power/Assign^train/wc1/Adam/Assign^train/wc1/Adam_1/Assign^train/wd1/Adam/Assign^train/wd1/Adam_1/Assign^train/wo/Adam/Assign^train/wo/Adam_1/Assign^train/bc1/Adam/Assign^train/bc1/Adam_1/Assign^train/bd1/Adam/Assign^train/bd1/Adam_1/Assign^train/bo/Adam/Assign^train/bo/Adam_1/Assign
X
HistogramSummary/tagConst*
dtype0*
_output_shapes
: *
valueB	 Bwc1
e
HistogramSummaryHistogramSummaryHistogramSummary/tagwc1/read*
T0*
_output_shapes
: 
Z
HistogramSummary_1/tagConst*
dtype0*
_output_shapes
: *
valueB	 Bwd1
i
HistogramSummary_1HistogramSummaryHistogramSummary_1/tagwd1/read*
T0*
_output_shapes
: 
Y
HistogramSummary_2/tagConst*
dtype0*
_output_shapes
: *
value
B Bwo
h
HistogramSummary_2HistogramSummaryHistogramSummary_2/tagwo/read*
T0*
_output_shapes
: 
Z
HistogramSummary_3/tagConst*
dtype0*
_output_shapes
: *
valueB	 Bbc1
i
HistogramSummary_3HistogramSummaryHistogramSummary_3/tagbc1/read*
T0*
_output_shapes
: 
Z
HistogramSummary_4/tagConst*
dtype0*
_output_shapes
: *
valueB	 Bbd1
i
HistogramSummary_4HistogramSummaryHistogramSummary_4/tagbd1/read*
T0*
_output_shapes
: 
Y
HistogramSummary_5/tagConst*
dtype0*
_output_shapes
: *
value
B Bbo
h
HistogramSummary_5HistogramSummaryHistogramSummary_5/tagbo/read*
T0*
_output_shapes
: 
�
MergeSummary/MergeSummaryMergeSummaryloss/ScalarSummaryaccuracy/ScalarSummaryHistogramSummaryHistogramSummary_1HistogramSummary_2HistogramSummary_3HistogramSummary_4HistogramSummary_5*
N*
_output_shapes
: "�BD�       Wca	C=)Xo�A*�@

loss�p@

accuracy�3�<
�
wc1*�	   ��꓿   @w0�?      r@!  �@n�?)��[��}�?2�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82�ji6�9���.��>�?�s���O�ʗ�����Zr[v��I��P=��>�?�s��>�FF�G ?��[�?1��a˲?�S�F !?�[^:��"?U�4@@�$?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?���#@?�!�A?�T���C?a�$��{E?
����G?IcD���L?k�1^�sO?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�������:�              @      @      @      @      $@      @       @       @      @      @      @      @       @       @      @      @      �?      @      @      @       @      �?      �?      @      @      @      �?      @       @      @              �?       @      �?              �?               @      �?      �?               @              �?              �?              �?              �?              �?              �?      �?               @              �?      �?              �?              �?       @              �?              �?       @       @      �?      @              �?      @       @      @      @      @      @      @       @      $@      @       @       @      $@      @      @       @       @      @      @      $@      @      @      @        
�
wd1*�	   ��z��   ��z�?      A!xrc0���)IfcY�A@2�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%���������m!#���
�%W��K���7��[#=�؏��f^��`{>�����~>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�������:�             ��@     ��@     �@    �c�@    ���@    ���@    ��@     ��@    ���@    ��@     �@    �O�@     ;�@     �@     �@     Ľ@     �@     J�@     `�@     h�@     /�@     ��@     ��@     |�@     .�@     ڨ@     z�@     `�@     V�@     n�@     ��@     L�@     �@     �@     l�@     ؓ@     ��@     P�@     �@     h�@      �@     ��@     0�@     `�@     (�@      }@     p{@     �x@      u@      u@     �q@     �q@     @p@      l@      h@     �h@      f@     �b@      ^@     �\@     `b@     �Z@     �W@     @V@     @S@     �U@     �R@     �M@     �J@      M@      D@     �C@      <@      B@      @@      ?@      ?@      1@      4@      @@      3@      4@      7@      (@      .@      &@      (@      &@      &@      $@      @      &@      @       @       @       @       @      @       @      @               @      @      @      @              �?      @      @       @              �?      �?               @              �?              �?               @      �?      �?      �?      �?              �?               @               @      �?              �?      �?      �?      �?      @      �?      @      @      �?      @      @      @              @      @      �?      @       @       @      "@      @      0@      $@      &@      *@      5@      1@      0@      8@      ;@      ;@      <@      ?@      D@      B@      F@      E@     �H@     �P@     @R@     �P@     @T@      [@      Z@     �Y@     @]@     �a@     �_@     �`@     `f@     `h@      j@     �n@     �k@     �p@     @t@     �s@     �v@     �y@     z@     �}@     ��@     ��@     �@      �@     �@     ��@     x�@     X�@     �@     |�@     l�@     ��@     |�@     ��@     \�@     x�@     ��@     f�@     ��@     �@     ��@     F�@     	�@     9�@     ��@     ��@     ��@     ��@     ��@     <�@    �+�@     D�@    �a�@    ���@    �-�@    ���@    �[�@    �O�@     ��@    ��@     ��@    ���@     7�@     >�@     Ȳ@        
�
wo*�	   `lv��   �c�?     ��@! XVQl��?)W�i��s�?2�	^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������1��a˲���[��O�ʗ�����Zr[v��})�l a��ߊ4F��5�"�g���0�6�/n��K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>})�l a�>pz�w�7�>��Zr[v�>O�ʗ��>1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�������:�	              ;@      A@     �D@     �O@     �H@      I@     @P@     �P@     �P@     �Q@      N@     @Q@      P@      I@     �D@      A@      K@      E@     �A@     �E@      @@      5@      @@      ;@      4@      4@      .@      1@      &@      (@      .@      (@      (@      @      $@       @      @      @       @       @      @      @      @       @               @      @      @      @       @      @      @      �?              �?              �?              �?      �?      �?      �?              �?              �?              �?               @              �?              �?      �?              �?              �?      �?      �?              �?              �?              �?              �?              �?       @              �?      @      @       @      �?      @      @      @       @       @      @       @      @       @      @      @      @      @       @      @      @       @       @      $@      &@      @      0@      .@      @      7@      .@      2@      8@      7@      8@      9@      A@      >@      F@     �A@      >@      H@     �D@      K@     �L@      L@     �O@     @Q@     @R@     �Q@      P@     @Q@     @Q@     �F@     �C@     �H@      ;@        
D
bc1*=      @@2        �-���q=�������:              @@        
D
bd1*=      P@2        �-���q=�������:              P@        
C
bo*=     �E@2        �-���q=�������:             �E@        ~�/�H!      o��F	�c�Xo�A*�B

loss��p@

accuracy��k=
�
wc1*�	   �x���   @�Ք?      r@!   ����?)�[N\T��?2�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C���%�V6��u�w74���82���VlQ.��7Kaa+�1��a˲���[��E��a�W�>�ѩ�-�>f�ʜ�7
?>h�'�?x?�x�?��82?�u�w74?uܬ�@8?��%>��:?d�\D�X=?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?�������:�              @      @      @      @      @       @      @      "@      @      "@       @      @      @      @      �?      @      @      @              @      @      @      @       @      @      �?               @      �?              �?      �?      �?      @      �?              �?               @      �?              �?              �?              �?              �?      �?               @              �?      �?              �?      �?              �?      �?       @      �?              �?      @      @       @       @      �?              @       @      @       @      @      @      (@      @      @      @      $@      @      @      @      @      @       @      @       @      @      @      @      @      �?        
�
wd1*�	   ��}��    �{�?      A!�"�kXL��)�qG&��A@2��"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L����|�~���MZ��K���u��gr��R%������39W$:���X$�z��
�}����T�L<��u��6
��K���7��BvŐ�r�ہkVl�p�w&���qa>�����0c>f^��`{>�����~>�
�%W�>���m!#�>�4[_>��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?�������:�             Pw@     ��@     R�@     ��@    �K�@      �@    ���@    ���@     ��@     ��@     ��@     �@    �y�@    ���@    �6�@     ߿@     �@     ׺@     ��@     ��@     ��@     #�@     ;�@     װ@     ��@     �@     Ĩ@     �@     ԣ@     ��@     t�@     ��@     ��@     ��@     H�@     ��@     ��@     ��@     ��@     P�@     ��@     Ј@     @�@     ��@     p�@     �@     �~@     @{@      x@     `x@     �u@     �r@     pq@     �p@     �i@      h@     �i@     @b@      c@     �e@     @]@     @_@     @\@     �Z@      U@     �T@     @P@      K@      Q@      I@      K@      J@      D@      B@      A@     �B@      :@      :@      0@      &@      8@      5@      9@      &@      &@      .@      (@      &@       @      &@      ,@      @      "@      �?       @      @      @      @      @      @      @      @      �?      �?      �?      �?      �?      �?       @              �?      @              �?              �?               @      �?              �?              �?              �?              �?      �?               @      �?      �?              �?      @               @       @       @              @      @      @       @      @      @      @      @      @       @      @      @      �?       @      .@      &@      &@      3@      1@      *@      $@      5@      4@      5@      6@      >@      B@     �@@      <@     �A@     �G@     �K@     �H@     �I@     �F@     �I@     �Q@      T@     �V@      W@      ]@     �[@      ^@     �a@     �c@     �e@      i@     �j@      j@     `m@     �q@     �s@     �t@     �v@     `y@     �|@     X�@      �@     P�@     �@     ��@     ��@     ��@     @�@     �@     ��@     �@     ��@     ��@     �@     4�@     �@     
�@     �@     
�@     ��@     ʩ@     P�@     8�@     �@     �@     ��@     ��@     ��@     ��@     ӻ@     ~�@     )�@    �'�@     �@     ]�@    �E�@    �%�@     <�@     H�@    ���@     ��@     ��@     ��@    �J�@     H�@     ߱@     @x@        
�
wo*�	   ��H��   ��P�?     ��@!  ��yϿ).;����?2��"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�U�4@@�$��[^:��"��S�F !���ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'�������6�]���1��a˲���Zr[v��I��P=��pz�w�7���5�L�����]����I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?�������:�              @      :@      B@      I@      K@     �J@     �J@     @Q@     �Q@      M@     @Q@     �S@     �L@     �J@     �F@      L@      H@      G@     �H@     �C@      B@      <@      5@      8@      9@      ,@      3@      *@      5@      6@      ,@      (@      1@      ,@      0@      @      "@      $@       @      "@       @       @      @      �?              �?      �?      @      @      �?              �?      �?              @      �?      �?      �?              �?              �?      �?               @      �?              �?              �?      �?              �?              �?              �?              �?      @       @      �?       @      �?              �?      @      @      @      �?      @      @      @       @      @       @      @      @      @      @      @      @      "@      @      @      0@       @      ,@      (@      (@      *@      3@      &@      2@      3@      6@      >@     �F@      ;@     �D@     �A@      ?@     �H@     �D@     �H@     �E@      R@     @Q@     �P@      R@     �P@     �N@      S@      J@     �G@      D@      D@      ;@      �?        
�
bc1*�	   �kaP�   @�`P?      @@!   ���^�)���WA ?2@nK���LQ�k�1^�sO�IcD���L��qU���I?IcD���L?k�1^�sO?nK���LQ?�������:@              ,@      @              �?              ,@        
�
bd1*�	   �aP�   `�aP?      P@!   �m�e?)p
,�~?2`nK���LQ�k�1^�sO�IcD���L��qU���I�
����G��T���C��!�A��qU���I?IcD���L?k�1^�sO?nK���LQ?�������:`              :@      @              �?              �?              �?       @      >@        
�
bo*�	   @;bP�   �DbP?     �E@!   <pn��)��f��?2(nK���LQ�k�1^�sO�k�1^�sO?nK���LQ?�������:(              :@              1@        �[n�"      |a]�	-y�Yo�A*�E

loss�Zp@

accuracy�p=
�
wc1*�	   ����   @=��?      r@!  (��m�?)_�"�?2��"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�k�1^�sO�IcD���L��qU���I�a�$��{E��T���C����#@�d�\D�X=�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�+A�F�&�U�4@@�$�x?�x��>h�'��I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%>��:?d�\D�X=?�!�A?�T���C?
����G?�qU���I?IcD���L?k�1^�sO?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?�������:�              �?       @      @       @      @      @      @       @      @      "@       @      @      @      @      @      @       @       @       @      @      @              @      @      @      �?      @       @      �?      �?       @               @      �?              �?              �?              �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?       @      �?              �?      �?       @      @      �?      @      @      @       @      �?      @      @      @      @      @       @      @      @      "@      @      @      @      @      @      $@      @      $@      @       @      @       @        
�
wd1*�	   ��+��   �^w�?      A! ���m@@)��= B@2��"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ���|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#��u��6
��K���7��Fixі�W���x��U�ڿ�ɓ�i>=�.^ol>�H5�8�t>�i����v>T�L<�>��z!�?�>��ӤP��>�
�%W�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?�������:�             h�@     i�@     ι@     6�@     )�@     ��@     ��@    ���@    ���@     i�@    ���@    ��@    ���@     +�@     ��@     ]�@     "�@     }�@     "�@     ޶@     �@     �@     x�@     T�@     l�@     ,�@     ��@     Ц@     ^�@     ��@     ��@     @�@     ��@     ܙ@     ��@     ��@     ܓ@     ��@     0�@     �@     (�@     ��@     ��@     0�@     ��@     0@     H�@     �{@     �y@     �w@     �s@     �r@     `p@     �m@     �l@      g@     `f@     �d@     �a@     �c@      ^@     �\@      Z@     �V@     @U@     �Q@     �U@     @P@     �N@     �K@      E@      I@     �G@      >@     �A@      <@      ;@      9@      3@      1@      4@      2@      7@      1@      ,@       @      &@       @      @       @      @       @      @       @      @      @      �?       @       @      @      @      @      �?      @      @      �?      @               @      �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?       @              @       @       @               @              �?      @      @      @      @       @      @      @      @      @      @      @      @      @      @       @      @      @      (@      1@      @      3@      4@      ,@      5@      6@      1@      5@      1@      7@      @@     �@@      7@      =@      C@      M@     �G@     �P@      L@      Q@     �P@      Q@      X@     �T@     @Y@     �]@     �a@     �\@      f@     �e@     �g@     @j@     �k@     �n@     @p@      s@     0u@      w@     @y@     �w@     ~@     ��@     X�@     8�@     І@     h�@     ȉ@     ȋ@     T�@     ��@     �@     ��@     |�@     X�@     ��@     �@     ��@     ��@     ~�@     b�@     �@     p�@     H�@     ð@     ��@     �@     I�@     ��@     #�@     @�@     ޽@    �$�@    �H�@    ���@    ���@     �@     Q�@     ��@    ���@     q�@    �]�@    ���@     �@    �V�@     �@     ��@     ��@        
�
wo*�	   ����   �X�?     ��@!  d;���)�@vw4�?2��"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��5�i}1���d�r�x?�x����[���FF�G ���(��澢f����>�?�s��>�FF�G ?��[�?1��a˲?6�]��?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?�������:�              &@      <@      A@     �M@      O@     �D@     �P@      L@     @Q@     �Q@     @R@      M@     �P@      L@     �K@      M@     �E@      H@      G@      A@      C@      ;@      3@      2@      ;@      0@      8@      8@      3@      5@      1@      &@      0@      @      @       @      ,@      (@      @      @      "@      �?      @      @      @       @       @      @      @      @       @      �?      �?      �?      �?      �?               @              �?       @              �?              �?              �?      �?      �?       @              �?       @      �?      @      �?       @               @       @       @              @      @       @       @      @      @      @      @      @      @       @      @       @      @      "@      &@      @      .@       @      (@      ,@      2@      &@      1@      1@      8@      9@      >@      :@      ;@      C@      E@      D@     �E@      H@     �@@     �K@     �Q@     �P@     �Q@     �L@      P@     �R@     �Q@     �F@      H@      D@     �C@      8@      @        
�
bc1*�	   ���[�    ��]?      @@!   �Z&�?)\ǺN?2��m9�H�[���bB�SY�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=�uܬ�@8���%�V6��u�w74���82���bȬ�0��!�A?�T���C?��bB�SY?�m9�H�[?E��{��^?�������:�               @              �?      �?              �?              �?              �?      �?               @              �?      @      �?      �?              �?              �?      *@        
�
bd1*�	   `�[[�   ��``?      P@!   �J�r?)V���l'?2��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS��vV�R9?��ڋ?�7Kaa+?��VlQ.?��bȬ�0?��82?uܬ�@8?��%>��:?
����G?�qU���I?�m9�H�[?E��{��^?�l�P�`?�������:�              :@      @              �?              �?               @               @              �?              �?               @      3@        
�
bo*�	   �=b`�   �4b`?     �E@!   �`n��)��AS�&?2(�l�P�`�E��{��^�E��{��^?�l�P�`?�������:(              :@              1@        �Z�"      |Wk�	j\Zo�A*�D

loss �o@

accuracyӼc=
�
wc1*�	    H��    ���?      r@!  �2aM�?)��Y�?2��"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C���VlQ.��7Kaa+���ڋ��vV�R9���[���FF�G ��[^:��"?U�4@@�$?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?�������:�               @      �?      @      @      @      @      @      @      @      @       @      @      @       @      @      @      @      @      @       @      @      @      @              @       @      @       @      �?      �?      �?      �?      �?       @       @              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?       @              �?              �?      �?       @       @              �?       @      @      @      @      @      @      @      @      @      @      &@      @      @      $@      @      @      @      @      @       @       @      @       @      @      @        
�
wd1*�	   �%���   @Ns�?      A!8��i�9Y@)���VbB@2��"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������]������|�~���MZ��K��R%������39W$:����4[_>������m!#��u��6
��K���7��f^��`{�E'�/��x���x��U�H��'ϱS��z��6��so쩾4���Ő�;F>��8"uH>�H5�8�t>�i����v>��ӤP��>�
�%W�>�4[_>��>
�}���>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�������:�             @�@     f�@     8�@     0�@     ��@     ��@    ���@    �#�@     ��@    �X�@    ���@    �}�@     ��@    ���@     ��@     ��@     ��@     Ⱥ@     ��@     ��@     ��@     Գ@     ��@     +�@     ȭ@     .�@     ̨@     ��@     r�@     �@     ��@     8�@     ț@     �@     <�@     ��@     @�@     ��@     (�@     ��@      �@     x�@     P�@     @�@     H�@     0~@     �}@      {@      y@     `x@     �u@     �r@     @o@      n@      m@      m@     �g@      h@     �b@     ``@     @^@      [@     @]@     @[@     �R@     @U@     �S@     @P@     �I@      G@     �N@      F@      H@      7@      ;@      B@      8@      >@      8@      5@      0@      0@      $@      $@      ,@      *@      $@      "@       @      @      @      @      $@      @      �?      @       @      @      @      @       @      @      @       @               @               @      �?               @               @              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?      �?       @      �?      �?      �?      �?      @      @      @      @      @      @      @      @      $@      *@      "@      @      @      @      @      *@      (@      0@      2@       @      @      5@      5@      7@      7@      <@     �D@     �@@      ;@      F@      K@      H@     �G@     �Q@     �M@     �P@     �S@      Y@      V@     �\@     �Z@     @a@     �_@      e@     `c@     `j@      g@     �l@      l@     �n@     �s@     �u@     0u@     �x@     {@     �~@     X�@     ��@     ��@     �@     x�@     X�@     ��@     X�@     l�@     ��@     t�@     l�@     ��@     ԛ@     ԟ@     <�@     h�@     ��@     ��@     @�@     ʫ@     n�@     ��@     c�@     I�@     +�@     ��@     z�@     �@     н@    ��@     R�@     H�@     ��@     3�@     e�@    ���@    �(�@    ���@     ��@     J�@     ��@     ��@     R�@     f�@     ��@     �f@        
�
wo*�	    s���    ���?     ��@!  $O��)����?2��"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1�x?�x��>h�'���ߊ4F��h���`��uE���⾮��%ᾢFF�G ?��[�?6�]��?����?f�ʜ�7
?>h�'�?��d�r?�5�i}1?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�������:�              3@      :@      B@      P@     �M@      N@      M@      L@      P@     @U@      L@      N@      Q@     �P@      I@     �M@     �F@     �G@     �A@      E@      8@      8@      <@      <@      @@      4@      :@      *@      4@      (@      0@      2@      1@      @      @      @      "@      "@      �?      "@      @       @      @      @      @      @      @      @      �?      @      @              �?      �?               @      �?       @      �?      �?              �?              �?              �?              �?              �?              �?              �?               @              �?       @      �?      �?      �?               @      @      @      @      @      @      �?      @      @      @      $@      &@      @      $@      @      @       @      1@      $@      0@      .@      0@      ,@      8@      6@      8@      3@      9@      ;@      4@      A@      =@      I@      G@     �E@      N@     �I@     �K@     �P@     �P@      O@      K@     �T@     �K@      L@      G@     �E@      >@      6@      (@      �?        
�
bc1*�	    ��b�    ��e?      @@!  �AHe�?)1�h��?2����%��b��l�P�`�<DKc��T��lDZrS��qU���I�
����G��u�w74���82��vV�R9��T7���O�ʗ��>>�?�s��>�[^:��"?U�4@@�$?��VlQ.?��bȬ�0?��82?�u�w74?��%>��:?d�\D�X=?���#@?�!�A?nK���LQ?�lDZrS?���%��b?5Ucv0ed?Tw��Nof?�������:�               @              �?              �?              �?              �?              �?              �?              �?              �?              @      @      �?              �?              �?      *@        
�
bd1*�	    ��a�   ���h?      P@!   ����?)4M*�C7?2����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�uܬ�@8���%�V6��u�w74���82���%>��:?d�\D�X=?a�$��{E?
����G?�qU���I?�lDZrS?<DKc��T?���%��b?5Ucv0ed?Tw��Nof?P}���h?�������:�              :@      @              �?               @              �?              �?              �?      �?              �?              �?      @      4@        
�
bo*�	    ]�h�   ��h?     �E@!   �~���).��аY9?2(P}���h�Tw��Nof�Tw��Nof?P}���h?�������:(              :@              1@        >k���"      !5Za	��[o�A*�E

loss�/o@

accuracy�ZS=
�
wc1*�	   `����    �f�?      r@!  P�J�?)�7��p��?2��"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G��!�A����#@���%>��:�uܬ�@8���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&��S�F !�ji6�9���.����ڋ�+A�F�&?I�I�)�(?�!�A?�T���C?
����G?�qU���I?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�������:�               @      �?      �?      @      @      @       @      @      @       @      @      @      @      @      @      @       @      @      @      @      @              �?      @      @      �?      @      �?              �?      @      @       @       @      �?              �?              �?              �?      �?              �?              �?              �?              �?              �?               @              �?      �?      �?      �?      @       @              �?      @              @       @      @      @      @      @      @       @      @      "@      $@      @      &@      @      @       @      @      "@      @      &@      @      @      @      �?        
� 
wd1*� 	   �Z ��    �o�?      A!�{��>�g@)�
�L�B@2�}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g����u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���u��gr��R%������39W$:���.��fc���X$�z�����m!#���
�%W����ӤP���u��6
��K���7��E'�/��x��i����v��H5�8�t�Fixі�W���x��U��H5�8�t>�i����v>E'�/��x>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�������:�             �R@     ��@     Ȯ@     ��@     K�@     R�@     ��@    �?�@    ���@     ��@    �:�@    �O�@    ���@     ��@     M�@    ���@     ��@     4�@     $�@     ��@     ��@     ��@     �@     ��@     w�@     h�@     Ʃ@     ��@     Z�@     ��@     ��@     ��@     ��@     d�@     �@     ��@     �@     ��@     (�@     @�@     8�@     Њ@     H�@     ��@     @�@     ��@     ��@     �}@     0y@     py@     �v@     �s@     `q@     �o@     �p@     �l@     `f@     `f@     @j@      _@     �b@     @`@      `@      Y@      Z@     �T@      V@     �O@     @P@     �P@      L@      J@      G@      G@     �@@      ;@      A@      6@      =@      7@      :@      0@      1@      7@      0@      @       @      @       @       @      "@      &@      @       @      @      @      @      @      �?      @      @      @              �?       @      @       @              �?              �?              �?      �?      �?      �?              �?      �?               @              �?       @              �?              �?      �?              �?      �?              �?      �?      �?              @              �?              @      �?      @      �?              @              @      @       @      �?      @      @      @      @       @      @      (@      &@      $@      @      "@      "@      "@      ,@      @      "@      0@      (@      0@      5@      9@      8@     �A@      A@      <@     �D@      E@     �E@     �G@      N@     �M@     @Q@      O@     �Q@     @V@      Z@     @V@      Z@      a@     �`@     �c@     �e@     �f@      h@     @m@     @l@     0p@     0t@     pt@     @u@     �w@     �|@     �|@     �@     ��@     ��@     P�@     `�@     h�@     ��@     (�@     ��@     ��@      �@     �@     ,�@     p�@     ��@     ��@     >�@     R�@     2�@     :�@     �@     L�@     v�@     ��@     w�@     	�@     ��@     ��@     ��@     ۽@     ��@    ���@    ���@     ��@    �F�@     p�@    ��@    �Y�@    ���@    ���@    ���@     _�@    �R�@     ޼@     �@     ��@     ��@        
�
wo*�	   @�r��   @ї?     ��@!  {����)�?����?2�}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1�f�ʜ�7
��������[���FF�G �>�?�s���O�ʗ���>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�������:�               @      6@      ;@     �E@      P@      K@     �Q@     @P@      H@     �R@      N@     �R@      O@      Q@     �P@     �M@     �E@     �L@     �A@      E@      8@      A@      A@      2@      A@      :@      :@      5@      ,@      4@      .@      (@      ,@      ,@      &@      @      "@      @      "@      @      @      @      @      �?      @      @       @      @      @      �?      �?       @      �?       @       @              �?       @              �?      �?              �?              �?              �?              �?              �?              �?              �?      �?               @      �?      �?              �?       @      �?       @               @      @       @      @      @      �?      @       @       @      @      @      "@      @      @      @      "@      @      $@      &@      &@      0@      3@      4@      1@      0@      1@      9@      3@      ,@      1@      A@      7@     �@@      B@     �D@     �E@      M@      J@      K@      I@      P@     �L@     �M@     �R@     @P@      K@     �M@      F@      G@      5@      9@      5@      �?        
�
bc1*�	   `,[f�    �l?      @@!   �pa�?)8����(?2�Tw��Nof�5Ucv0ed�nK���LQ�k�1^�sO�f�ʜ�7
������uܬ�@8?��%>��:?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?�m9�H�[?E��{��^?P}���h?ߤ�(g%k?�N�W�m?�������:�               @              �?              �?              �?              �?              �?               @               @      @       @              �?              @       @        
�
bd1*�	   @�ee�   ��bp?      P@!   ^��?)E��O0C?2�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^�IcD���L��qU���I�
����G��[^:��"��S�F !�<DKc��T?ܗ�SsW?E��{��^?�l�P�`?P}���h?ߤ�(g%k?�N�W�m?;8�clp?�������:�              9@      @              �?               @      �?              �?               @              �?              �?      @      6@        
�
bo*�	   �)bp�   ��ap?     �E@!   �on��)�}��̈́F?2(;8�clp��N�W�m��N�W�m?;8�clp?�������:(              :@              1@        ����<#      %�n 	�B�[o�A*�F

loss%n@

accuracy��D=
�
wc1*�	   �}ᕿ   ��J�?      r@!  �����?)�n���l�?2��"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�IcD���L��qU���I�a�$��{E��T���C�d�\D�X=���%>��:�uܬ�@8���bȬ�0���VlQ.��7Kaa+��.����ڋ�>h�'��f�ʜ�7
�����?f�ʜ�7
?>h�'�?x?�x�?�u�w74?��%�V6?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?k�1^�sO?nK���LQ?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�������:�               @              �?      @      @      @      @      @      @      @      @      @      @      @       @       @      @      @       @      @      �?      @      @       @              �?       @       @       @      �?      @              �?              �?              �?       @              �?      �?              �?              �?              �?              �?              �?               @              �?       @              �?              �?              �?               @      @      �?      @      @              @       @       @      @      @      @      @      @      @       @      &@      @      &@       @      @      "@      @      $@      @      "@       @      @      @        
�
wd1*�	    �}��   �xn�?      A!�	ks�r@)�O�)TC@2�}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d�����?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����z!�?��T�L<���'v�V,>7'_��+/>/�p`B>�`�}6D>ہkVl�p>BvŐ�r>u��6
�>T�L<�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�������:�              l@     ��@     b�@     ��@     A�@     ��@    ���@    ���@    ���@     ��@    ���@    ���@    ���@     W�@     ��@    �.�@     ��@     Q�@     �@     q�@     =�@     ��@     ��@     ]�@     `�@     `�@     �@     "�@     ԥ@     ��@     ��@     `�@     �@     ��@     d�@     �@     �@     ��@     0�@     ��@     ��@     ��@     ��@     ��@     �@      �@     8�@     �}@     �x@     `w@     �v@     Pt@     pr@     �p@     `m@     �p@      i@     @j@     @e@     �d@      `@     �[@     �\@      [@     �T@     �Q@     @R@     �P@     @Q@      O@      P@     �D@      G@      A@      A@      G@      =@      @@      .@      6@      7@      4@      1@      8@      (@      (@      ,@      ,@      &@      @      (@      @      @       @      @      $@       @      "@      @       @      @      @              @               @      �?      �?       @              @      �?       @              �?       @      �?               @              �?              �?              �?              �?              �?              �?       @       @              �?              �?      @       @       @       @      @      @      @      @      @      @      @      @      &@      @      @      &@      *@      @      "@      "@      ,@      "@      (@      3@      1@      6@      =@      1@      ?@      <@      =@      B@     �A@     �H@     �C@     �L@     �F@     �P@     �R@      N@     @U@     @T@     @Y@     �\@      `@     �a@     �a@     �c@      i@      j@     �j@     �m@     pp@     �r@     �t@     �u@     �y@     �{@     P~@     p�@     ��@     �@     ��@     `�@     0�@     ��@     �@     @�@     ,�@     X�@     ܖ@     ̙@     �@     T�@     ��@     ��@     ң@     �@     ��@     D�@     ��@     ��@     ��@     e�@     ��@     ѷ@     ��@     H�@     ܽ@     ��@     ��@     ��@    ���@    �x�@     ��@     '�@    ���@    �B�@     ��@    ���@     �@    �R�@     ��@     ��@     ܧ@     t�@      B@        
�
wo*�	   ��M��   `���?     ��@!  �=���)��ʲ<�?2�	}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.���T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������>�?�s���O�ʗ�����Zr[v���MZ��K���u��gr������>豪}0ڰ>�uE����>�f����>pz�w�7�>I��P=�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�������:�	               @      8@     �@@      H@      K@     �Q@      O@     �N@      O@     �O@     �P@     @Q@     �R@     �L@     �M@     @Q@     �H@      D@     �H@      >@      ;@     �@@     �@@      A@      ;@      2@      :@      1@      *@      (@      .@      *@      ,@      ,@      *@      @      @      "@      @      "@      @      @      @      @      �?      @      �?      @      @       @      �?      @      @              �?      �?      �?              �?              �?      �?      �?       @               @      �?              �?              �?              �?              �?              �?      �?      �?              �?              �?              �?              �?      �?      �?      @       @       @      �?      �?      @      �?      @       @               @      @      @      @      @      @      @      @      @      $@      @      @      @      @      @      0@      $@      3@      2@      $@      0@      &@      7@      4@      2@      ;@      8@     �@@      C@      C@      C@      C@     �K@      J@      I@      P@      E@      O@     �P@     �Q@      L@     �K@     �L@     �H@     �E@      8@      7@      4@      @        
�
bc1*�	    �i�   ��q?      @@!  @���?)�4��'4?2�ߤ�(g%k�P}���h�Tw��Nof��T���C��!�A�a�$��{E?
����G?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?�N�W�m?;8�clp?uWy��r?�������:�              �?      �?              �?              �?              �?              �?              �?      �?       @       @      @              �?              @      "@        
�
bd1*�	   �wUh�   ࡂt?      P@!  �"*�?)��ֵ��L?2�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�<DKc��T��lDZrS�nK���LQ�a�$��{E��T���C��l�P�`?���%��b?5Ucv0ed?Tw��Nof?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?�������:�              ;@       @              �?               @      �?              �?               @              �?              �?      �?      2@      @        
�
bo*�	   �Fzt�    Iyt?     �E@!   ��
��)E��\��Q?2(&b՞
�u�hyO�s�hyO�s?&b՞
�u?�������:(              :@              1@        ���<#      %�n 	�#�\o�A*�F

losswl@

accuracy��[=
�
wc1*�	   ��.��   �B3�?      r@!  ��KU�?)�N�9�?2��"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���d�r?�5�i}1?ji6�9�?�S�F !?�7Kaa+?��VlQ.?��82?�u�w74?��%�V6?uܬ�@8?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�������:�               @              �?      �?      @      @      @      @      @       @       @      @       @      @      $@      @      @      �?      @       @      @      @               @       @      @      @      �?      �?      �?               @      �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?               @              @      �?      @      �?      �?      �?      �?      �?      @       @      @       @      @       @      @      @       @      @      "@      "@      $@      @      ,@      @      @      "@       @       @      $@      "@      @      @      �?        
�
wd1*�	   ��͗�   @'o�?      A!���~�|z@)�
���C@2�}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K��R%������39W$:���.��fc���X$�z��
�}����u��6
��K���7��E'�/��x��i����v�7'_��+/>_"s�$1>w`f���n>ہkVl�p>T�L<�>��z!�?�>.��fc��>39W$:��>R%�����>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�������:�             �u@     ��@     �@     d�@     �@    ��@    �=�@    ���@     ��@    ��@    �Y�@     ��@    ���@    ���@    ���@     �@     ��@     6�@     ��@     '�@     z�@     ��@     1�@     4�@     į@     �@     �@     T�@     \�@     "�@     ��@     v�@      �@     ��@     �@     ��@     ��@     (�@     �@     �@     �@     ��@     ��@     p�@     X�@     Ѐ@     P~@     �z@      y@     @x@     �t@     �s@     �q@      q@     �l@     @j@     �h@     �e@     �f@     �a@     �^@     �a@      ]@     �X@      Q@      N@     @T@      N@     �K@      P@     �G@      E@     �A@     �G@     �D@      ?@      =@      E@      7@      2@      1@      3@      5@      *@      @      (@      @      ,@      "@      @       @      @      @      "@      @      @      @      @      @      @      @      �?       @              @      �?      �?       @      �?       @              @      �?               @              �?      �?              �?              �?              �?              �?              �?               @      �?              �?      �?      �?      @      �?              @       @       @       @              �?      �?      @              @      @      @      @       @      @      @      "@      *@       @      $@      ,@      $@      &@      2@      ,@      3@      5@      :@      <@      <@     �C@      8@      >@     �C@      H@     �A@      O@     �H@     �Q@     �F@      S@     �V@     �X@     @]@     @Z@     @^@     ``@     �a@      i@      h@     @k@     �i@      m@     �p@     �r@     pt@      v@      v@     �{@     �{@     Ѐ@     ��@     �@     P�@     ��@     ��@     x�@     �@     ��@     P�@     ��@     ؗ@     T�@     ��@     4�@     L�@     Ƣ@     ��@     �@      �@     �@     ¬@     %�@     �@     ��@     �@     1�@     �@     p�@     ��@     Ϳ@    ���@     ��@    �&�@     ��@    �2�@     ��@     Q�@    ���@     I�@     ��@     ��@     F�@     ľ@     �@     Ƭ@     �@      |@        
�
wo*�	    �0��   ���?     ��@! �R�z�)�E��7��?2�	��<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��f�ʜ�7
�������FF�G �>�?�s���O�ʗ���pz�w�7��})�l a�h���`�8K�ߝ뾋h���`�>�ߊ4F��>})�l a�>pz�w�7�>��Zr[v�>O�ʗ��>��[�?1��a˲?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�������:�	              �?      .@      <@      @@      J@      O@      Q@     �O@      Q@     �P@      M@     @R@      L@      U@      K@      L@     �P@      I@     �E@      ?@     �C@     �@@      ?@      ?@      >@      3@      5@      5@      0@      3@      &@      5@      ,@      *@      $@      $@      (@      @      @      @      @      @       @      @       @      @      @      @      �?      @      @      �?       @      @      �?       @               @      �?       @      �?              �?      @              �?               @       @              �?              �?              �?              �?              �?              �?               @      �?              �?               @              �?      �?      �?              @      �?               @      �?      @      @      �?      "@      @      @      @       @       @      @      �?      $@      @      (@       @      "@      @      &@      (@      1@      (@      6@      ,@      1@      =@      5@     �C@     �@@     �A@      A@     �@@      F@     �F@      K@     �G@     �N@     �H@     @P@      I@     @Q@      N@     �M@     �I@      I@      E@      <@      3@      8@      (@      �?        
�
bc1*�	    t^l�    ku?      @@!  ��~��?)��؅H�>?2��N�W�m�ߤ�(g%k�P}���h��ߊ4F��h���`�<DKc��T?ܗ�SsW?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?;8�clp?uWy��r?hyO�s?&b՞
�u?�������:�              �?      �?              �?              �?              �?      �?              @      �?      @       @              �?      @      "@        
�
bd1*�	    ��j�   �ܧx?      P@!   �*@�?)��k��S?2�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b���bB�SY�ܗ�SsW�nK���LQ�k�1^�sO�Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?�������:�              ;@       @              �?              @              �?               @              �?              �?      �?      3@      @        
�
bo*�	   �]�x�    Ȑx?     �E@!   �ު��)3���:Y?20o��5sz�*QH�x�&b՞
�u?*QH�x?o��5sz?�������:0              :@              �?      0@        ���ތ"      |Wk�	�p]o�A*�D

loss�pj@

accuracyΈR=
�
wc1*�	   `�r��   `�!�?      r@!  �6��?)T4K�6�?2��"�uԖ�^�S�������&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E���bȬ�0���VlQ.��S�F !�ji6�9���.����ڋ�ji6�9�?�S�F !?U�4@@�$?+A�F�&?��82?�u�w74?��%�V6?d�\D�X=?���#@?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�������:�               @               @      @      @      @      @      @      @      @       @      @      @       @      @      @       @      @      @       @      @       @       @      �?      @              @      �?       @      �?       @       @               @              �?              �?              �?              �?              �?              �?              �?      �?              �?               @       @      �?      �?       @      �?       @       @       @       @       @      �?      �?       @      @       @      @      @      @      @      "@      "@      @      &@      &@      @      (@      @       @      "@      $@      "@       @       @      @       @        
�
wd1*�	   ��O��   �:r�?      A!�$�qx�@)�n���D@2�}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���u��gr��R%������.��fc���X$�z���4[_>������m!#���
�%W����ӤP�����z!�?�������~�f^��`{�H��'ϱS>��x��U>E'�/��x>f^��`{>u��6
�>T�L<�>���m!#�>�4[_>��>
�}���>39W$:��>R%�����>�u��gr�>�MZ��K�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�������:�             �~@     D�@     D�@     ��@     �@     !�@     ��@     �@    �*�@    ���@    �|�@    ���@    � �@     ��@    ��@     ��@     �@     x�@     �@     )�@     C�@     �@     Ų@     ��@     :�@     z�@     n�@     ��@     �@     b�@     ��@     "�@     ��@     �@     <�@     4�@     ��@     T�@     �@     h�@     ��@     ��@     ؆@     Є@     Ȃ@      �@     �~@     �z@     px@     px@     v@     `r@     0r@      q@     `k@     �j@      j@     @e@      f@      a@     �_@     �\@     �_@     �\@     @U@     �P@      P@      T@     �O@     �L@     �G@      L@     �F@      D@      <@      =@      @@      @@      =@      .@      $@      (@      3@      2@      $@      "@      *@      $@       @      @       @       @      @      @      �?      @       @      @      @      @      @       @      @      �?      �?       @      �?       @              �?      @      �?              �?              @              �?               @      �?               @              �?              �?              �?              @      �?              �?              @              �?       @      �?      �?      �?       @      @       @      �?              �?      @      @      @       @      @      @      "@      @       @      (@      @      0@      $@      *@      (@      6@      .@      (@      9@      2@      5@      9@      6@      9@      =@      9@     �A@      A@      D@     �I@     @P@      Q@     �S@     @T@      V@      Z@      [@      _@     �\@     `a@     �`@     �d@      f@      h@     �g@     @k@     Pp@     �r@     �s@     �u@     `x@     �x@     @|@     x�@     ؀@     ��@     ��@     ؆@      �@     ��@     0�@     ��@     p�@     ��@     ��@     ܘ@     ��@     ��@     �@     ��@     ��@     R�@     ��@     Ȫ@     ��@     y�@     ޱ@     N�@     ��@     @�@     r�@     �@     �@    �4�@     ��@    �l�@     �@     �@    �d�@     �@     !�@     Z�@    ���@    ���@    ��@    ��@    ��@     �@     ��@     Ʀ@     $�@        
�
wo*�	   �)��   �"��?     ��@! �R����)(-�7���?2�	��<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��T7����5�i}1���d�r������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v���uE���⾮��%�G&�$�>�*��ڽ>;�"�q�>['�?��>>�?�s��>�FF�G ?6�]��?����?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�������:�	              @      3@     �A@     �A@      J@     �P@     @P@      R@     �Q@      N@      R@     @P@      O@     �S@     �M@     �H@     �K@     �H@      E@      A@     �@@      A@      C@      @@      0@      7@      :@      5@      4@      6@      *@      1@      4@      ,@      &@      @      @      @      @      @      @       @      @      @      @      @       @      @      @       @      @      �?      �?      �?      �?      �?              @      �?              �?      �?              �?               @      �?              �?      �?              �?              �?              �?              �?              �?              @       @               @      �?              �?              �?      �?       @      @      �?       @      @      @      @      @      @      �?       @       @      @      "@      &@      @      "@      *@      .@      .@      $@      0@      $@      :@      4@      8@      =@      B@      4@      =@     �A@      @@      I@     �H@     �H@      K@      I@      N@     �M@      J@     �H@     �O@     �Q@      B@     �L@      E@      B@      3@      7@      .@      @        
�
bc1*�	   ���n�    �y?      @@!  �� �?)0C���IF?2�;8�clp��N�W�m�ߤ�(g%k��T���C?a�$��{E?E��{��^?�l�P�`?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?�������:�              �?      �?              �?              �?               @              @      �?      @      �?              �?      @       @        
�
bd1*�	    �m�   ���|?      P@!   bĴ?)�#�#�Z?2��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed�E��{��^��m9�H�[�ܗ�SsW�<DKc��T��N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?�������:�              :@      @              �?              @              �?               @              �?              �?      @      6@        
�
bo*�	   �<�|�   @��|?     �E@!   �+��)��ߥa?2(���T}�o��5sz�o��5sz?���T}?�������:(              :@              1@        �eYl"      ~�#	6�.^o�A	*�D

loss�h@

accuracy��C=
�
wc1*�	   `D���   ���?      r@!  ĩvg�?)�Z
틡?2��"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���VlQ.��7Kaa+�I�I�)�(�1��a˲?6�]��?�vV�R9?��ڋ?�.�?uܬ�@8?��%>��:?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�������:�              �?      �?               @              @      @      @      @      @      @      �?      �?      @      @       @      @      @      @      @      @      �?      @              �?      @      @               @              �?      @              �?      �?      �?               @               @       @              �?              �?      �?              �?              �?      �?      �?              �?       @      �?      �?      �?              @      @      �?      �?      @       @      @       @       @      @       @      @      @      @      $@      $@      "@      &@      &@      $@      &@      @      &@       @      $@      $@      $@      @      @        
�
wd1*�	   ��   ��x�?      A!����
�@)J&~7�E@2�}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ������;9��R���5�L�����]������|�~���MZ��K��R%������39W$:���.��fc���
�}�����4[_>������m!#����z!�?��T�L<��p��Dp�@>/�p`B>�����~>[#=�؏�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>���]���>�5�L�>;9��R�>���?�ګ>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�             P�@      �@     �@     ƴ@     �@     V�@     ��@     ��@     ��@     ��@    ��@     ��@     ��@     ��@    �-�@     ��@     �@     >�@     ��@     /�@     ô@     ��@     �@     ��@     H�@     n�@     D�@     P�@     ڤ@     ܢ@     T�@     d�@     ȝ@     `�@     H�@     t�@     ��@     |�@     �@     ؍@     H�@     Љ@     8�@      �@     ��@     @�@     �~@      |@     �y@     0v@     �t@     �r@      p@     @o@     �l@      j@      g@     �d@      c@     ``@     �Z@     �_@     �]@     @]@     �W@     �S@      N@     �V@      N@     �M@     �G@      M@     �@@      C@      =@      =@      3@      1@      7@      6@      =@      2@      .@      0@      $@      &@      .@      &@      @       @      &@       @      "@      @       @      @      @      @      @      @      @      @      �?      @      @              �?              �?      �?      �?       @              �?      �?              �?      �?              �?              �?              �?              �?      �?      �?               @              �?              �?      �?              �?               @              @      @      @      @      @       @      @       @      @       @      �?      @      @      @      @      @      ,@      "@      $@      .@       @      .@      *@      ,@      .@      .@      5@      =@      ?@      5@      @@      =@      D@     �G@      F@      G@     �K@     �V@     �Q@     �P@     @R@     @V@      Y@     @]@      ]@     �[@      `@     `c@     �d@     �c@     �k@     `m@     �o@     �q@     �s@     ps@     0x@     �z@     |@      �@     X�@     ��@     x�@     ��@      �@     h�@     �@     ��@     ��@     ��@     �@     4�@     ��@     ĝ@     ��@     ̡@     v�@     .�@     ܧ@     ��@     �@     9�@     Q�@     ��@     д@     !�@     `�@     ں@     �@     ֿ@     \�@     ��@     ��@     ��@     ��@    �]�@     ��@     ��@    ���@    �b�@     +�@     ��@     ��@     (�@     �@     "�@     D�@     �b@        
�
wo*�	    ���   �n��?     ��@! �v#A^�)O�!2c��?2���<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��5�i}1���d�r�>�?�s���O�ʗ���pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�������:�              @      8@     �A@      E@      P@     �L@      Q@     @T@      O@     �Q@     @R@     @P@     �P@      Q@     �J@     �J@     �J@      E@      D@      @@     �A@     �D@     �@@      7@      =@      @@      :@      3@      2@      ,@      $@      5@      1@      &@      $@      &@       @      @      @      @      @      @      @      �?      @      @      @      �?      @      �?       @       @       @              �?      @              �?       @              �?              �?               @              �?      �?              �?      �?      �?              �?              �?      �?      �?      @              �?      �?       @              �?      @       @      @      �?              �?              @       @      @      @       @      @      "@      @       @      @      &@      @      0@      &@      .@      2@      @      (@      5@      4@      2@      7@      ;@      ;@      7@      7@     �B@     �@@     �G@      K@      L@      K@      H@      L@     �J@      M@      E@      M@      O@      I@     �I@      E@     �D@      6@      3@      3@      $@        
�
bc1*�	   ��xp�    ��|?      @@!  ��5h�?)�����N?2�uWy��r�;8�clp��N�W�m�ߤ�(g%k��lDZrS?<DKc��T?5Ucv0ed?Tw��Nof?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?�������:�              �?              �?              �?              �?               @       @       @      @       @              @      &@        
�
bd1*�	   �uo�    b��?      P@!   �uP�?)����a?2�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?�������:�              9@      @      �?      �?               @      �?              �?               @              �?              �?      1@      "@        
�
bo*�	   @V[��   ��^�?     �E@!   ���)��*@f?28����=���>	� �����T}����T}?>	� �?����=��?�������:8              9@      �?               @      .@        �~�#      %�\9	���^o�A
*�F

loss��e@

accuracy8�B=
�
wc1*�	    7㖿    �?      r@!  ��E�?)9�]%�?2�}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I��T���C��!�A����#@�d�\D�X=��u�w74���82���bȬ�0���VlQ.�I�I�)�(�+A�F�&���bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?              �?               @               @      @      @      @      @      @      @      �?      @      �?      @      @       @       @      @      @      @      @       @       @               @       @              @              @      �?              �?      �?      �?              �?               @              �?              �?      �?      �?      �?              �?              �?      �?      @               @       @              �?              @               @      �?      �?      @      @      �?      @      @      @       @      @      @      @      @      @      *@       @      *@      &@      *@      &@      @      *@       @      $@      $@       @      @      �?        
�
wd1*�	   �Q���    ���?      A!�F�m��@)pa���F@2���<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������X$�z��
�}������ӤP�����z!�?����u}��\�4�j�6Z�E'�/��x>f^��`{>��ӤP��>�
�%W�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?     ��@     "�@     v�@     Դ@     w�@     F�@     ο@     (�@     G�@    �L�@    ��@     ��@     ��@    �t�@     l�@     ��@     ��@     m�@     _�@     ��@     �@     &�@     ��@     ��@     �@     2�@     Ƨ@     P�@     x�@     ��@     �@     ��@     P�@     ��@     И@     ��@     ��@     `�@     �@     0�@     P�@     P�@      �@     ��@      �@     h�@      }@     |@     `x@      w@     u@     `s@     �p@      q@      l@     `l@     �f@     �b@      b@      b@     @^@      _@      `@     @Z@      Y@     �X@      T@     �Q@     �P@     �O@     �E@      E@      C@     �C@     �C@      7@     �C@      ?@      4@      (@      5@      0@      .@      1@      (@      &@      *@      $@      @      @      @      @       @       @      @      @       @       @      @      @      @      @      @      @       @      �?              �?      �?      @               @              �?      @              �?              �?              �?              �?              �?              �?      @              �?      �?      �?              �?              �?      �?       @      �?              @       @       @       @      �?      �?      @      @      @      @      �?      @      @      @      @      @      *@      $@      (@      &@      @      .@      2@      0@      4@      0@      <@      0@      @@      ;@     �F@      ?@     �D@      9@      E@      J@      Q@     @P@     �Q@     �S@     �R@     �Y@     �Y@     �V@     @^@     �`@     �b@      d@     �g@     `g@     �l@     @p@     �p@     r@     Pu@     pw@     �x@      ~@      �@     h�@     `�@     H�@     p�@     ؈@     ��@     ��@     �@     ��@     ܒ@     ��@     l�@     X�@     ��@     �@     ��@     B�@     ̤@     ��@     (�@     N�@     ��@     �@     ��@     %�@     Ķ@     ��@     �@     B�@     J�@     ��@     s�@     ��@    ��@    ���@     ��@    ���@    �K�@    �_�@     0�@     M�@    �+�@     ��@     b�@     N�@     <�@     4�@     ��@        
�
wo*�	   �e���    Z�?     ��@! ���	�)I�Zc�?2�	�v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲�>�?�s���O�ʗ���8K�ߝ�a�Ϭ(龮��%ᾙѩ�-߾E��a�WܾG&�$��5�"�g�����~]�[�>��>M|K�>�h���`�>�ߊ4F��>x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�	              �?      (@      A@     �@@     �F@      O@     �M@     �S@     �R@     @Q@     @S@     @R@      P@     �N@      P@     �L@     �I@      C@      A@     �D@     �E@     �D@     �D@     �A@      8@      9@      ;@      8@      5@      ,@      1@      3@      &@      (@      0@      $@      *@      $@      @      @       @      @       @      @      �?       @      �?       @      @       @              �?       @       @       @      �?              �?       @       @               @              �?               @              �?      �?              �?              �?              �?      �?              �?              �?              �?              �?      �?      �?      @      �?              @              �?              @      @      �?       @       @      @      �?       @      @      @      @      @      @      @      @       @      @       @      &@      *@      .@      @      @      .@      5@      "@      1@      1@      0@      @@      ?@      8@      8@      7@     �I@      G@     �E@     �N@      I@     �K@      J@      J@     �I@      L@      H@     �N@      G@      I@     �D@     �G@      ;@      2@      4@      ,@      �?        
�
bc1*�	    lq�   ��f�?      @@!  @����?)��j�P�T?2�uWy��r�;8�clp��N�W�m�E��{��^?�l�P�`?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?�������:�              �?      �?              �?              �?               @      �?      @       @              �?      @      @        
�
bd1*�	   @�`p�    ���?      P@!   ��?) ��-ce?2�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^�&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?�������:�              ;@       @              �?               @      �?      �?               @      �?              �?      "@      1@        
�
bo*�	   @c_��    �i�?     �E@!   �����)D����k?2@���J�\������=���>	� �����T}?>	� �?����=��?���J�\�?�������:@              9@      �?              �?      �?      .@        ���_\#      $�@�	�"`o�A*�F

loss�c@

accuracy�DX=
�
wc1*�	   `x��   @��?      r@!  8��H�?)�����?2�}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY��lDZrS�nK���LQ��qU���I�
����G�a�$��{E�ji6�9���.����ڋ�x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�[^:��"?U�4@@�$?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?              �?              �?      �?              @      @      @      @      @      @      @      @       @      �?      �?      @      @       @      �?              @      �?      @      @       @      @               @              �?       @              �?       @              �?      �?      �?              �?              �?              �?      �?               @      �?      �?      �?              �?              @      �?      �?      @       @       @              @      �?       @      @      @       @      @      @       @      @      @      @      @      *@      "@      .@      (@      .@      @      $@      $@      (@      &@      &@      @      @        
� 
wd1*� 	    y$��   �t��?      A!7J'��@){!�RH@2���<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:����4[_>������m!#����ӤP�����z!�?��T�L<��w&���qa�d�V�_�������M>28���FP>BvŐ�r>�H5�8�t>�i����v>E'�/��x>�����~>[#=�؏�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�������:�              �?     H�@     ��@      �@     ̴@     -�@     ��@     x�@     ��@    �Y�@    ���@     ��@      �@     Z�@    ��@     ��@     ��@     �@     �@     �@     ε@     T�@     i�@     i�@     �@     ��@     ��@     �@     ��@     �@     
�@     ��@     ̞@     H�@     P�@     t�@     ؔ@     �@     P�@     d�@     ��@     h�@     ��@     �@     ��@     ��@     p�@      @     @y@     �y@     �u@     �s@     �q@     `m@     �k@     @l@     `j@     @g@     �f@      b@     @_@      ^@     �\@     �Z@     �X@     @U@     �X@      Q@     �P@     �J@     �L@     �L@      F@     �C@      =@     �@@      :@      8@      0@     �@@      ;@      1@      0@      .@      4@      .@      @      &@      &@      @      "@      @       @      @      @              @      �?      �?      @      �?      @      @      �?      @       @       @      �?       @       @       @      �?       @              �?              �?               @              �?      �?              �?              �?              �?      �?      �?               @              �?              �?              �?      �?      �?              �?              �?       @      �?       @       @      �?              �?       @       @      @       @      @      @       @      @      @       @      "@      @       @       @      $@      @      *@      @      $@      &@      3@      7@      4@      .@      1@      5@      >@      =@      8@      C@      A@     �D@     �J@      O@     �I@     �Q@     �Q@     �X@     �U@     �Y@      X@      ]@      \@     @b@      a@     `d@      h@      i@     @h@     �l@     �q@     �q@     �u@     �u@     �v@     p{@      |@     ��@     Ȃ@     0�@     ��@     ��@     ��@     Ѝ@     p�@     ��@     ԓ@     0�@     ̗@     ��@     ,�@     ԟ@     ��@     v�@     ��@     ��@     Ĩ@     :�@     ή@     �@     ��@     �@     ��@     з@     D�@     E�@     ��@     �@     ]�@     ��@    �	�@    ���@    ���@     p�@     ��@    ���@    �4�@     �@    ��@    �]�@    �8�@     -�@     ��@     ҫ@     t�@      @        
�
wo*�	    ����   �I��?     ��@! �ݨ���)gn[�3�?2�	�v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
�1��a˲���[��;�"�qʾ
�/eq
ȾK+�E���>jqs&\��>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>��Zr[v�>O�ʗ��>>�?�s��>f�ʜ�7
?>h�'�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�	              @      3@     �@@     �B@     �N@     �N@      K@     @R@     �T@     �S@     �Q@     �S@      K@     �N@      P@      J@      G@     �D@      @@      H@     �H@     �A@     �B@      @@      ;@      8@      ;@      4@      0@      3@      *@      .@      0@      1@      *@      $@      $@      "@      ,@      @      @      @      $@      "@      �?      �?      @      �?       @      @      @       @       @       @              �?      �?      @       @               @      �?               @      �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?      �?              �?      �?      �?              �?      �?       @      �?      �?      @      �?              �?      @      @       @      @      @      @      @      @      (@      *@      &@      @      *@      @      &@      1@      $@      .@      3@      5@     �@@      .@      8@      A@      <@      C@     �G@     �J@      H@     �D@      Q@      K@      I@      I@     �J@      J@     �G@      M@     �L@      A@     �D@     �@@      8@      2@      2@       @        
�
bc1*�	   @UEr�   �	h�?      @@!  �l6`�?)7��'o�Z?2�hyO�s�uWy��r�;8�clp��N�W�m�5Ucv0ed?Tw��Nof?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?�������:�              �?              �?              �?              �?               @       @       @       @      �?      @      "@        
�
bd1*�	   �W&q�   �:��?      P@!   �]&�?),w���0j?2�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?�������:�              9@      @      �?      �?               @      �?      �?               @      �?              @      5@        
�
bo*�	   �
\��   �3t�?     �E@!   us��)���{�p?2P-Ա�L�����J�\������=���>	� ��o��5sz?���T}?����=��?���J�\�?-Ա�L�?�������:P              9@              �?              �?              @      *@        ,�̪�#      M�Na	|��`o�A*�G

loss"tb@

accuracy�xi=
�
wc1*�	   ��<��   �F�?      r@!  ���%@)��w�`#�?2�}Y�4j���"�uԖ��Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��[^:��"��S�F !�ji6�9��+A�F�&?I�I�)�(?��bȬ�0?��82?�u�w74?d�\D�X=?���#@?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?              �?      �?      �?               @      @       @      @      @      @      @      @      @      �?       @       @      �?       @      �?      @      @      �?      �?              �?       @              �?       @       @      �?               @      �?      �?      �?              �?      �?              �?              �?      �?              �?              �?              @      �?      @      �?               @      @      �?              @      @      @       @              @      @      @      @      @      @      @      @      @       @      @      0@      $@      .@      *@      0@       @      (@      $@      (@      $@       @      @        
�!
wd1*�!	    ~��   ���?      A![tHjz��@)S��I@2���<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>����
�%W����ӤP�����z!�?��T�L<��f^��`{�E'�/��x�ڿ�ɓ�i�:�AC)8g������0c�w&���qa���u}��\�4�j�6Z���f��p>�i
�k>4��evk'>���<�)>w&���qa>�����0c>cR�k�e>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�������:�              �?     T�@     �@     (�@     	�@     ط@     ��@     ��@     ��@    ���@     ��@     7�@     ��@     ��@     �@     ��@     d�@     ۹@     @�@     z�@     U�@     g�@     �@     ް@     l�@     t�@     �@     ئ@     t�@     
�@     ,�@     |�@     ��@     ��@     x�@     ��@     �@     ��@     ��@     h�@      �@     Љ@     (�@      �@     0�@     Ё@     �~@      |@     @z@     �x@     pt@     �t@      r@     �o@     �l@     �j@     `f@     �d@      b@     �d@      `@     �\@     �[@     @V@     �Z@      R@     @P@     @Q@     �J@      P@      E@      E@     �F@      F@      A@      6@      7@      >@      2@      @@      0@      ,@      1@      2@      2@      (@      3@      "@      @      ,@      @      $@      @      @      @      �?       @      @      @      @       @      @       @      @      �?      �?               @      �?      �?       @       @              �?      �?      �?       @               @      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?      @              �?              �?      �?       @       @      �?      �?               @              �?      @       @       @      @      @       @      @      @      @       @      "@      @       @      @      @      @       @      $@      "@      0@      1@      *@      (@      1@      (@      2@      4@      @@      4@     �@@      9@     �G@     �C@     �F@      I@      J@     �K@      T@      S@      W@      X@     @Z@     �X@      `@     @b@     �b@     `d@      d@     �i@     @h@     �j@     pp@      q@     pt@     �u@     `v@     p{@     �|@     @@     ؀@     ��@     ��@     8�@     H�@     (�@     <�@     @�@      �@     t�@     ��@     ��@     p�@     �@     ��@     r�@     <�@     Φ@     ��@     6�@     ʮ@     U�@     ӱ@     ҳ@     ��@     ��@     �@     U�@     ��@     8�@    ��@     i�@     >�@     ��@     ��@     R�@     Y�@    �z�@     �@    �7�@     g�@     	�@     ��@     �@     ^�@     :�@     ��@      q@        
�
wo*�	   �����   ����?     ��@! �z�,D�)�m�H�E�?2�	�v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !��.����ڋ��vV�R9���d�r�x?�x��f�ʜ�7
������6�]���1��a˲���[���FF�G �O�ʗ�����Zr[v������ž�XQ�þG&�$��5�"�g���O�ʗ��>>�?�s��>6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�	              $@      4@     �B@     �B@      Q@      N@     �O@     �R@     �S@     �U@     @Q@      R@      M@      M@     �K@      K@     �F@     �H@     �C@      C@      F@     �C@      ?@     �A@      8@      8@      =@      6@      .@      0@      0@      4@      (@      0@      *@       @      $@      @      @      @      @      "@      @      @      @       @      @      @              @      �?              �?      @      �?      @              �?      �?              �?              �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?      �?              �?               @              �?               @       @      �?      �?      �?       @      @      �?      @       @       @      @      @      @       @       @      @      "@      "@      $@      @      $@      *@       @       @      0@      5@      &@      6@      5@      8@      :@      @@      @@     �B@     �E@      F@     �J@     �F@     �L@      N@     �I@      M@      E@     �K@     �E@     �I@      M@     �D@     �B@     �A@      ?@      .@      5@      @        
�
bc1*�	   `s�    
o�?      @@!  �-�?)h�}�%a?2xhyO�s�uWy��r�;8�clp�P}���h?ߤ�(g%k?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?�������:x              �?      �?              �?              �?               @      @      "@              @      "@        
�
bd1*�	   �y�q�   �&��?      P@!   a�4�?)^V&�(\o?2xuWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b����T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?�������:x              ;@       @              �?              @      �?               @      �?              @      6@        
�
bo*�	   �hL��   ���?     �E@!   |X繿)�,<�es?2`eiS�m��-Ա�L�����J�\������=���>	� ��o��5sz?���T}?����=��?���J�\�?-Ա�L�?eiS�m�?�������:`              7@       @              �?              �?              �?      @      (@        9*���"      Kމ	/��ao�A*�E

loss�@b@

accuracy�~{=
�
wc1*�	   ��b��   `O�?      r@!  �H�@)��?{�0�?2�}Y�4j���"�uԖ��Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�k�1^�sO�IcD���L��qU���I��!�A����#@�d�\D�X=���%>��:��u�w74���82���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$�['�?�;;�"�qʾ��d�r?�5�i}1?+A�F�&?I�I�)�(?�u�w74?��%�V6?a�$��{E?
����G?�qU���I?k�1^�sO?nK���LQ?�lDZrS?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�������:�              �?              �?      �?               @      �?       @      �?      @      @      @      @      @      @      �?      @       @       @      �?               @      �?              @      @      �?      �?              �?       @              �?               @              �?              �?      �?              �?              �?              �?               @              �?              �?       @              �?       @              �?      �?      @      @              @       @      @      @      @      @      @      @      @      @      @      @      @      @      "@       @      .@      *@      0@      1@      $@      $@      $@      *@      (@      "@      @      �?        
�
wd1*�	    +��    "<�?      A!�)F��@)�r$��\K@2���<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��39W$:���.��fc���X$�z��
�}�����
�%W����ӤP���f^��`{�E'�/��x��i����v������0c�w&���qa�d�V�_���u}��\�[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�4[_>��>
�}���>���]���>�5�L�>;9��R�>���?�ګ>����>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�������:�               @     d�@     ��@     8�@     .�@     �@     N�@     !�@     2�@     ?�@    �`�@     n�@     7�@     �@     �@     ��@     ��@     ��@     -�@     �@     д@     �@     ��@     3�@     ,�@     "�@     ��@     ��@     �@     b�@     ��@     ��@     ��@     ��@     |�@     ��@     0�@     ̐@     L�@     `�@     �@     �@     0�@     h�@     ��@     ��@     �~@     �z@     px@     �v@     �t@     s@     �p@     �m@     `k@     �h@     �h@     �c@     �a@     �d@     �`@     �^@     �[@     �Z@     �T@      R@      P@      M@     �K@     �H@     �I@      J@     �D@     �L@      @@      1@      ?@      ;@      6@      2@      4@      1@      ,@      &@      0@      "@      @      @      &@      @       @      *@      &@      @      $@      @      @      @      @       @      @       @      �?       @      @      �?      �?      @      �?              �?      �?      �?      �?      �?               @              �?              �?               @       @              �?              �?              �?              �?              �?              �?              �?      @       @       @               @       @      �?       @      @      @      @      �?      @      @      @      @      $@       @       @      @      @      (@      .@      &@      (@      1@      0@       @      7@      5@      0@      0@      9@      6@      <@      >@     �A@     �C@      A@     �G@      N@      O@      Q@     �R@      R@      S@     @U@      [@     @]@     `a@     �`@     �c@      c@     �g@     �h@     �i@     Pq@     �p@      u@     �s@     �v@     @z@     |@     �~@     h�@     ��@     ��@     (�@     Ј@     p�@      �@     ��@     �@     �@     ��@     ؗ@     @�@     (�@     ��@     ��@     ��@     �@     r�@     Ī@     >�@     J�@     ڱ@     �@     1�@      �@     ��@     �@     ,�@    ���@     ��@     ��@     ��@     ��@    ���@    �r�@    ��@     %�@    ���@    �!�@    �)�@     :�@    ��@     �@     �@     <�@     ��@     0�@        
�
wo*�	    ���   ����?     ��@! ���f�)���W=-�?2��/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��6�]���1��a˲���[���ѩ�-�>���%�>��[�?1��a˲?����?f�ʜ�7
?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?      &@      4@      F@      H@      O@      R@     �J@      U@      U@     �Q@     �R@     @Q@      J@      M@      H@     �O@      L@      D@     �B@     �D@      H@     �C@      5@     �@@      A@      >@      8@      1@      ,@      1@      *@      ,@      0@      1@      &@      (@      @      @      @      "@      @      @      @      �?      @      @       @      @      @               @      �?      �?      @      �?       @              �?      �?      �?              �?              �?              �?      �?              �?              �?              �?              �?      �?              @      �?       @               @              @       @      @       @               @       @       @      @      @       @       @      @       @      @      @      @      $@       @      (@      (@      ,@      $@      &@      *@      0@      1@      ,@      7@      9@      4@      >@     �@@      B@      F@      H@      H@      F@      M@     �K@      K@      K@      I@     �E@      I@      L@     �I@      E@     �B@     �B@      =@      6@      4@      @        
�
bc1*�	    �s�   @%b�?      @@!  �����?)@oU�e?2xhyO�s�uWy��r�;8�clp��N�W�m?;8�clp?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?�������:x              �?      �?              �?              �?               @       @      @      @      @      "@        
�
bd1*�	   ��vr�   `Ϝ�?      P@!  �t�?)~�8�Pr?2xhyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed�>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�������:x              7@      @       @              �?      @      �?               @      �?              @      4@        
�
bo*�	   @*��    ���?     �E@!   �B��)͕ٷ v?2p#�+(�ŉ�eiS�m��-Ա�L�����J�\��>	� �����T}�*QH�x?o��5sz?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�������:p              2@      @      �?              �?              �?              �?       @       @      &@        ��\y�"      }C		�[bo�A*�E

lossn c@

accuracy�~{=
�
wc1*�	   @9���    t��?      r@! �����@)��!���?2�}Y�4j���"�uԖ��Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ��qU���I�
����G��!�A����#@���VlQ.��7Kaa+��[^:��"��S�F !��ߊ4F��>})�l a�>6�]��?����?��d�r?�5�i}1?I�I�)�(?�7Kaa+?��82?�u�w74?��%�V6?uܬ�@8?�!�A?�T���C?a�$��{E?
����G?�qU���I?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�������:�              �?              �?      �?               @      �?       @      �?      @       @      @      @      @      @      @       @      @      �?      �?               @               @      �?      �?              @      �?       @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @               @       @              �?              �?       @              �?       @              �?      @      �?      @      @      @       @      @      @      @      @      @      @      @      @      @      $@      (@      *@      0@      .@      0@      "@      *@      (@      &@      (@      @       @        
�
wd1*�	   `칛�   �y��?      A![��X��@)6
)L@2��v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R�����]������|�~���MZ��K���u��gr��X$�z��
�}�������m!#���
�%W����ӤP�����Ő�;F>��8"uH>w`f���n>ہkVl�p>39W$:��>R%�����>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�������:�              �?       @     |�@     �@     ��@     W�@      �@     p�@     �@     ھ@    ��@    �)�@    ��@     @�@     ��@     �@     ׻@     r�@     ��@     O�@     d�@     �@     ��@     }�@     �@     2�@     (�@     ��@     ��@     �@     >�@     Π@     ��@     0�@     0�@     Ԗ@     l�@     �@     ��@     ��@     ��@     ��@     8�@     P�@     8�@     Ё@     @�@      |@     |@      w@      x@     �u@     �q@      r@     `n@     `i@     @j@     �f@     @d@      b@      b@     `b@     �W@     �]@     @S@     @S@      T@     @R@     �Q@     �O@      I@      F@      C@      9@     �C@     �B@      8@      7@      7@      6@      2@      *@      0@      7@      *@      .@      4@      (@      .@      @      @      @      &@      @      @      @      @      @      �?      @      @       @      @      @       @      �?      �?      �?      �?       @      �?              �?      �?       @               @              �?       @              �?              �?              �?              �?              �?              �?       @       @      @      �?      �?      �?      @       @               @      @      @      @      $@      @      @      "@       @      (@      @      "@      3@      0@      ,@      0@      *@      1@      4@      .@      >@      @@      6@      =@      5@     �D@     �@@      F@      J@      L@      Q@     �P@     @R@     �Q@     �R@     �W@     @Z@     �X@     �_@     @c@      b@      d@     �e@     �g@      l@     �p@     �p@     �s@     `s@     `u@     �z@     @}@     @     �@     ��@     (�@     p�@     X�@     ��@     H�@      �@      �@     ��@     \�@     ��@     L�@     ��@     ��@     ¡@     ��@     l�@     ^�@     P�@     L�@     ,�@     ��@     ��@     (�@     �@     K�@     }�@     ܽ@    ���@     ��@    ��@     ��@     _�@     ��@    �U�@     >�@     P�@     ��@     ��@     ��@     ��@    �4�@     ��@     C�@     w�@     ��@     |�@        
�
wo*�	   �    &��?     ��@! b�v�)��@���?2�	�/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7�����d�r�x?�x��>h�'��f�ʜ�7
�6�]���1��a˲���(��澢f����E��a�Wܾ�iD*L�پ�iD*L��>E��a�W�>pz�w�7�>I��P=�>>�?�s��>�FF�G ?1��a˲?6�]��?��d�r?�5�i}1?�T7��?��ڋ?�.�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�������:�	              @      $@      7@      G@     �G@     �P@     @R@      N@      U@     @T@     @S@      P@      Q@      K@      J@      O@     �L@     �H@      F@     �@@      F@      H@     �E@     �@@      ;@      5@      :@      .@      <@      3@      2@      2@      .@      "@       @      ,@      (@      @       @      @      @      @      @       @      �?      @      �?      @      @      @      @      �?      �?       @       @              @              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @      �?              �?              �?       @      �?      �?      �?      �?      �?       @       @       @      @      @      �?      @      @      &@              @      @      "@       @      .@      ,@      $@      $@      @      0@      (@      .@      3@      (@      ,@      3@     �A@      <@      =@     �D@      D@      G@      H@     �E@     �J@     �N@     �K@     �E@      K@     �F@      I@     �N@      H@     �D@      F@      ;@      ?@      >@      1@      @      �?        
�
bc1*�	    �Tt�   �t�?      @@!   �0L�?){/x���h?2�&b՞
�u�hyO�s�uWy��r�;8�clp�uWy��r?hyO�s?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�������:�              �?              �?              �?              �?              �?      @      @      @      @       @        
�
bd1*�	   �Qs�    �G�?      P@!  ��֕�?);!y)��t?2phyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�������:p              9@      @      �?      �?      �?      @               @      �?      �?      0@      $@        
�
bo*�	   ��    ���?     �E@!   V�e��)����x?2��7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����T}�o��5sz�&b՞
�u?*QH�x?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�������:�              @      4@       @              �?              �?              �?      �?      �?      �?      "@      @        ��Zhz"      X/�	P4co�A*�D

loss�^c@

accuracy!t=
�
wc1*�	    <���   �j͟?      r@! ��ƅ6@)X��ʘ-�?2�}Y�4j���"�uԖ��Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�k�1^�sO�IcD���L��!�A����#@�d�\D�X=�8K�ߝ�a�Ϭ(�6�]��?����?�vV�R9?��ڋ?�.�?ji6�9�?�u�w74?��%�V6?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�������:�              �?              �?      �?               @      �?       @      �?      @       @      @      @      @      @       @      @      @       @      �?      �?      �?              @              �?      �?       @      @              �?              �?      �?              �?              �?              �?              �?              �?              �?       @      �?               @      �?              �?      �?               @              �?       @               @      @       @      @      @      @      @      @       @      @      @      @      @       @      @      $@      $@      .@      ,@      2@      *@      "@      ,@      *@      &@      &@      @       @        
�
wd1*�	   �\|��   @*�?      A!p�����@)�D�,	M@2��v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]�����MZ��K���u��gr��R%������39W$:����
�%W����ӤP���:�AC)8g�cR�k�e�ہkVl�p>BvŐ�r>��z!�?�>��ӤP��>X$�z�>.��fc��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?�������:�              �?      @@     ��@     \�@     ð@     ��@     J�@     ��@     :�@     ��@     �@     "�@    � �@     -�@     �@     ܽ@     �@     /�@     ��@     �@     ,�@     6�@     ��@     3�@     ��@     Ϋ@     �@     ��@     ��@     ��@     ��@     ؠ@     4�@     ��@     �@     P�@     ��@     ��@     �@     ��@      �@     P�@     ��@     ��@     ��@     8�@     Ȁ@     �|@     0{@     �w@     `v@     �s@      q@     `q@     �m@     �h@     �g@     `h@     �d@      c@     @_@     �a@      ]@     �V@     @Y@     @W@     @S@     @P@      L@      M@     �I@      M@      G@     �D@      B@      ?@      ;@      2@      8@      2@      ,@      2@      1@      2@      1@      1@      @       @      &@      @      "@      @       @       @      @       @      @      @      @      @      @      @      @      @       @       @      �?      @               @       @       @      �?              �?      �?      �?              �?              �?              �?              �?              �?               @      �?      @      �?              �?      @              @               @      �?      @       @      @       @      @      @              @      @      @      @      @      @      $@      $@      4@      (@      (@      &@      @      ,@      ,@      0@      8@      8@      8@      6@      :@      <@      <@     �A@     �G@      H@      Q@     �L@      J@     @T@     @R@     @U@     �W@     �Y@     �^@      a@      a@     `a@     �d@      g@     �i@      j@     Pp@     `o@     �s@     pt@     `v@     Py@     �}@      @     x�@     p�@     ��@     0�@     �@      �@     ؎@     L�@     `�@     P�@     ��@     �@     ��@     ��@     &�@     V�@     L�@     �@     �@     �@     *�@     |�@     |�@     �@     ��@     ��@     %�@     ��@     ��@    ���@     \�@    ���@    ���@     %�@     ��@    �'�@    ��@    ��@     ��@     ��@    ���@    �!�@    ���@     7�@     7�@     �@     ��@     �@      <@        
�
wo*�	   ��E��   ໍ�?     ��@!  :Nh.�)R1V�R�?2��/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"�ji6�9���.����ڋ�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[����~]�[Ӿjqs&\�Ѿ�iD*L��>E��a�W�>})�l a�>pz�w�7�>f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�������:�              @      (@      <@     �G@      F@      S@     @P@     �R@     �S@     @R@      S@      O@     @P@     �M@      N@      I@      O@     �I@      >@      F@      G@     �F@     �F@      ?@      9@      ;@      7@      0@      3@      0@      6@      1@      .@      *@      ,@      @      "@       @      @      &@      @       @      @      @       @       @       @      @      @      @      �?       @      �?              �?      @       @              �?      �?              �?      �?       @              �?      �?              �?              �?              �?              �?               @      �?              �?              �?      �?       @              �?       @      �?      �?       @       @      �?      @       @       @      @      @      @      @      @      @      &@      &@      @      *@       @      &@      *@      .@      .@      6@      &@      (@      1@      2@      ?@      =@     �A@     �B@     �E@     �H@      B@      F@      K@     �M@     �H@     �H@      J@      G@      K@      N@     �H@      @@      J@      =@      =@      ;@      5@      @       @        
�
bc1*�	   �~�t�   �=J�?      @@!   N�U�?)m�`V�k?2p&b՞
�u�hyO�s�uWy��r�hyO�s?&b՞
�u?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�������:p              �?      �?              �?              �?               @      @      "@      @      &@        
�
bd1*�	    �s�    �Ë?      P@!   ���?)|Cu,�v?2phyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�������:p              :@      @              �?      @      �?              �?       @               @      9@        
�
bo*�	   `_���   �*��?     �E@!   FH��)ˋ�~�{?2��7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\��o��5sz�*QH�x�hyO�s?&b՞
�u?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?�������:�              3@      @              �?              �?              �?              �?       @              �?      �?      $@      �?        :��:#      _�/P	m��co�A*�F

loss��b@

accuracyڬz=
�
wc1*�	   �L���   �:��?      r@! @���@)\�|W�?2�}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�
����G�a�$��{E���VlQ.��7Kaa+��.����ڋ��vV�R9�E��a�Wܾ�iD*L�پ��bȬ�0?��82?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?a�$��{E?
����G?IcD���L?k�1^�sO?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�������:�              �?              �?              �?      �?      �?       @      �?       @      @       @      @      @      @       @       @      @       @      @              @               @      �?              �?      �?       @      �?       @              �?      �?              �?              �?              �?      �?              �?               @              �?       @      �?       @              �?              �?              �?      �?      �?              @      �?      @      @       @      @      @      �?      @      @       @      @      @      @      @      @      "@      "@      "@      .@      1@      2@      $@      "@      .@      *@      $@      &@      @       @        
� 
wd1*� 	   �=U��    C�?      A!8��:��@)�	�tM@2��v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}����[#=�؏�������~���u}��\�4�j�6Z������0c>cR�k�e>u��6
�>T�L<�>�4[_>��>
�}���>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?�������:�              �?      O@     ��@     ��@     �@     ͵@     ��@     ڻ@     ��@     b�@     �@    �#�@    �,�@     P�@     k�@     ٽ@     һ@     ��@     J�@     L�@     O�@     /�@     ��@     �@     ү@     ��@     ��@     ��@     <�@     �@     �@     F�@     �@     ��@     ��@     ��@     l�@     (�@     D�@     Џ@     ��@     Ј@     ȇ@     ��@     �@     ��@     ��@     @}@      {@     �w@     @w@     0u@     Pq@     `p@     �m@      i@     @i@     �b@     �d@      e@      b@     @[@      ^@     �X@     �[@     �U@     �R@      Q@      L@     �G@      N@      G@      K@      D@      D@      B@     �@@      ;@      4@      3@      8@      0@      0@      ,@      ,@      .@      $@      *@      @      $@      "@      @      @       @       @      @      @       @      @       @      @      �?      @      @      �?       @      @              �?      �?       @       @      �?       @              �?       @      �?              �?      �?              �?              �?              �?              �?              �?              �?               @      �?              �?      �?      �?               @              �?      �?      �?      �?      �?      @      @      @      @       @      @      @      �?      @      @      $@       @      @      *@       @      "@      ,@      *@       @      0@      3@      ,@      9@      1@      5@      >@      A@      D@     �E@      C@     �L@     �J@      N@     �Q@     �T@      M@     �V@     @W@     �Z@     �W@      _@      `@     �a@     @b@      h@      i@     @m@     �n@     ps@     �q@     �s@     �u@     �z@     p}@     �~@      �@     ��@     ��@     Ї@     ��@     ��@     ��@     ��@     p�@     �@     �@     ��@     l�@     @�@     4�@     ��@     �@     H�@     (�@     �@     ��@     '�@     !�@     ��@     �@     �@     �@     �@     ׽@     M�@     ��@     ��@     ��@     �@    ���@     �@    ��@    ���@    ���@     ��@    ���@     	�@     S�@    �C�@     �@     ��@     p�@     ��@      L@        
�
wo*�	   ��x��   @��?     ��@! p���)`Iw~��?2�	�/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]�����[���FF�G ��h���`�>�ߊ4F��>��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�������:�	               @      *@      =@      F@     �F@     �S@      O@      S@     �S@     �S@     @Q@     �N@      P@     @P@     �M@      K@     �G@      L@     �D@     �E@      F@      E@      B@     �@@      <@      ?@      9@      4@      (@      5@      4@      ,@      (@      *@      .@      @       @      (@      @      @      @      @      "@      @      @      @      �?      �?       @      �?              @       @      @              �?       @               @              �?      �?              �?      �?      �?              �?      �?              �?              �?              �?      �?      �?              �?               @      @      �?              �?      �?      @       @               @      @      �?      �?      �?       @       @       @       @      @       @      @      $@      @       @      "@      &@      @      (@      $@      *@      .@      1@      .@      $@      1@      8@      8@      7@      9@      B@     �A@     �E@     �J@      @@     �F@     �J@      N@     �I@      G@      K@     �E@     �M@      L@     �H@     �B@      H@      @@      >@      :@      2@      @       @        
�
bc1*�	   ��du�   ����?      @@!   ���?)9<���0m?2x&b՞
�u�hyO�s�uWy��r�hyO�s?&b՞
�u?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�������:x              �?      �?              �?              �?              �?      @      @      @      $@      @        
�
bd1*�	   �j�s�   ���?      P@!  ��y��?)�f�T2�x?2x&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?�������:x              5@      @       @              �?      @              �?      �?      �?      �?      2@       @        
�
bo*�	   `UI��   �Ӹ�?     �E@!   �n���)K�sv5}?2��#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\��*QH�x�&b՞
�u�hyO�s?&b՞
�u?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?�������:�              $@      &@      @              �?              �?              �?               @      �?              �?      �?      @      @        _V���$      z��P	靡do�A*�I

loss�4b@

accuracy�{r=
�
wc1*�	   ��ٗ�   `��?      r@!  +C�@)׿��,�?2�}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�<DKc��T��lDZrS�nK���LQ�a�$��{E��T���C��!�A����#@�uܬ�@8���%�V6��.����ڋ��iD*L�پ�_�T�l׾��[�?1��a˲?�.�?ji6�9�?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��82?�u�w74?��%>��:?d�\D�X=?�T���C?a�$��{E?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�������:�              �?              �?              �?      �?      �?       @      @              @      @      @      @      @      �?      @      @      @      �?       @      �?      �?      �?       @      �?      �?       @       @              �?      �?              �?              �?               @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?               @               @      �?      �?      @      @              @      @       @      @      @      @      @      @       @      @      "@      @       @       @      $@      0@      ,@      1@      (@      "@      (@      ,@      (@      "@      @       @        
� 
wd1*� 	   `�>��    *n�?      A!�_;����@)~#f���L@2��v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L����|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}������z!�?��T�L<�������~�f^��`{���Ő�;F��`�}6D�/�p`B���-�z�!�%������x��U>Fixі�W>�i����v>E'�/��x>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?�������:�               @      V@     ��@     ��@     F�@     ��@     ޸@     D�@     ��@     ��@    ���@    ��@     ��@     ��@     �@     �@     ��@     ĺ@     ��@     ��@     ��@     P�@     Ͳ@      �@     ��@     6�@     ��@     >�@     �@     @�@     b�@     &�@     P�@     p�@     p�@     ��@     l�@     ��@     T�@     ��@     ��@     (�@     ��@     ��@     ��@     ؁@     ��@     0|@     Py@     �x@     �t@     Pu@     �q@     �q@     �l@      m@     �i@      e@     �c@     `b@     �a@      a@     @[@     @Y@     @Y@     �S@     �S@      P@     �M@     �M@      P@      J@     �J@     �C@     �A@      @@      7@     �A@      9@      =@      4@      2@      3@      3@      .@      @      "@      "@      (@      (@       @      @      (@      @      (@      "@      @      @      @      @      @      @       @               @       @      @              �?       @      �?       @              �?       @       @               @              �?              �?              �?              �?      �?              �?              �?               @              �?              �?              �?              @      �?              @       @               @      �?              �?       @      @      @      @      �?      @      @      @      @      @      @      "@      $@      @       @      $@      @      *@      "@      $@      @      $@      (@      0@      .@      ,@      4@      2@      =@      ;@     �@@     �C@      :@     �E@      K@     �F@     �P@      P@     @U@     �S@     @T@     �T@     �X@     �\@     ``@      c@     �c@      c@     �e@     �j@     `i@     @p@     �q@     �t@     �s@     �t@     �x@     �|@     �@     Ё@     `�@     �@     x�@     Ȉ@     x�@     ��@     ��@     ��@     �@     ��@     h�@     ��@     �@     l�@     �@     £@     ��@     ��@     
�@     ��@     ί@     L�@     Ӳ@     �@     ��@     �@     P�@     ��@     A�@     p�@    ���@     V�@    �M�@     ��@    ��@     ��@    ���@     m�@    ��@    �\�@     ��@    ���@    ���@     #�@     �@     ��@     ��@     �P@        
�
wo*�	   @�נ�    \�?     ��@! ��Ȉ��)���׊�?2�	�uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1�x?�x��>h�'��f�ʜ�7
��FF�G �>�?�s���O�ʗ�����Zr[v��I��P=���h���`�8K�ߝ���(��澢f������(���>a�Ϭ(�>8K�ߝ�>�h���`�>>�?�s��>�FF�G ?��[�?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�������:�	              �?       @      0@      =@      D@     �H@     �R@      Q@     �P@      U@     �R@     �Q@      L@     �L@     �R@     �L@      L@      G@      L@      @@     �O@     �D@     �C@      :@      A@      =@      =@      ;@      5@      2@      6@      5@      ,@      (@      $@      $@      $@      $@      @       @      @      @      �?      @       @      @      @      �?      @      @              �?       @      �?      �?              �?      @      �?              �?              �?              @              �?      �?              �?              �?      �?              �?              �?              �?              �?              �?      �?              �?      �?              �?              �?              �?      �?              @              �?      �?      �?      �?      @      @       @               @      @      "@       @      @      @      ,@      @      @      "@      $@       @      @      $@      &@      3@      0@      .@      (@      *@      7@      9@      <@      ?@      9@      E@      F@     �G@     �C@      C@      L@     �N@     �G@     �H@      G@     �J@      L@      M@     �F@      D@      H@      A@      >@      ;@      0@      @       @        
�
bc1*�	   �m�u�    ~E�?      @@!  �T��?)F���8�m?2x&b՞
�u�hyO�s�uWy��r�hyO�s?&b՞
�u?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�������:x              �?      �?              �?              �?              �?      @      @      @      @      @        
�
bd1*�	   `�gt�   ��?      P@!  �����?)`?K�"�y?2x&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?�������:x              8@      @       @              @      �?              �?      �?      �?               @      3@        
�
bo*�	    ����   `�_�?     �E@!   �z��)���4/?2��#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\��hyO�s�uWy��r�hyO�s?&b՞
�u?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�������:�              .@      @      @              �?              �?              �?              �?       @              �?              �?      �?      "@      �?        ��Sz$      %�]�	�)[eo�A*�H

loss�4a@

accuracy�ew=
�
wc1*�	    ��   @��?      r@! ��,@)��>��!�?2�}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G��!�A����#@���%>��:�uܬ�@8���%�V6��u�w74���VlQ.��7Kaa+�I�I�)�(�+A�F�&������>
�/eq
�>I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?�!�A?�T���C?
����G?�qU���I?IcD���L?k�1^�sO?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�������:�              �?              �?               @               @      �?      @       @      @      @      @      @      @       @      @      @      �?       @       @      �?       @       @       @      �?      �?               @              �?              �?              �?      �?              �?              �?      �?      �?              �?      �?      �?              �?              �?              �?              �?              �?      �?      �?               @      �?      �?      �?      @      @       @      @      @       @      @      @      @      @       @      @       @       @      @       @      @       @      @      1@      *@      .@      ,@      (@      $@      &@      &@      *@       @      @       @        
� 
wd1*� 	   @�0��   `���?      A!(�͑�E�@)����L@2��/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n����n�����豪}0ڰ���������?�ګ��MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#��K���7��[#=�؏��BvŐ�r�ہkVl�p�6NK��2>�so쩾4>�����0c>cR�k�e>E'�/��x>f^��`{>[#=�؏�>K���7�>�
�%W�>���m!#�>�4[_>��>X$�z�>.��fc��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?�������:�              �?       @     �_@     ��@     8�@     �@     o�@     	�@     ��@     ��@     $�@    ���@    �q�@    ���@     �@     ��@     �@     j�@     %�@     �@     ��@     	�@     _�@     &�@     ��@     )�@     ��@     *�@     �@     ��@     Z�@     ��@     h�@     `�@     �@     ��@     <�@     �@     �@     h�@     t�@     ��@     �@     І@     ؅@     ��@     8�@     ��@     @}@     0{@     y@     �u@     `s@     `r@     r@     �m@     �l@     �h@     �f@      f@     @c@      `@     �_@     �V@     �Z@     �Y@     @T@     �R@     �R@     �Q@     �P@      P@      G@     �C@     �C@      H@      @@      @@      ?@      2@      3@      ;@      2@      .@      2@      1@      .@      ,@      @      &@      @      .@      @      @      @      @      "@      @      @       @      @       @      @      @               @               @              �?              @              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?               @      �?              �?               @              @      @      @      @      �?      �?      �?       @              @      @      @      @      @      @       @      "@      @      .@      @      *@      *@      $@      (@      ,@      3@      *@      3@      :@      :@      6@      8@      7@      ?@      @@      ?@     �A@      M@      H@     �M@      Q@      R@      Q@     �V@     �V@     @\@      `@     `a@     @c@     �e@      f@     �j@     �k@     @i@     q@     @q@      s@     `t@     �t@     �{@     @~@     0�@     p�@     ��@     X�@     ��@     p�@     ��@     ��@     �@     0�@     ��@     ��@     ��@     p�@     Ĝ@     `�@     �@     ��@     ��@     ��@     X�@     .�@     �@     ��@     �@     �@     Զ@     K�@     �@     :�@    �:�@     T�@    ��@     ^�@    �0�@    ���@    ���@    ���@    �q�@    ��@     ��@     ��@     �@    �(�@    �I�@     պ@     ��@     �@     ��@     �N@        
�
wo*�	   ��&��   @��?     ��@! �P��[�)�X�Go�?2�	�uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.���vV�R9��T7����5�i}1���d�r�x?�x��>h�'�������6�]���1��a˲���[���FF�G �pz�w�7��})�l a�XQ�þ��~��¾�XQ��>�����>��>M|K�>�_�T�l�>})�l a�>pz�w�7�>1��a˲?6�]��?>h�'�?x?�x�?��d�r?�5�i}1?�vV�R9?��ڋ?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�������:�	               @      @      0@      =@      D@      J@     @R@     �O@      Q@     @S@     �T@      N@     @Q@     �H@     @Q@     �O@     �N@     �G@     �G@     �E@     �H@      E@     �E@      ?@      D@      7@      7@      ;@      1@      6@      3@      9@      (@      3@      @      5@      *@      @      @      @      @      @       @       @      @      @       @       @      @       @      �?      �?       @      �?       @      �?      �?       @       @               @              �?              �?              �?              �?              �?       @               @              �?              �?              �?              �?              �?              �?      �?      �?               @               @              @       @      @      �?              �?      @       @       @       @      �?      @      @      @      @      "@      @      @       @      @       @      "@      "@      ,@      (@      3@      (@      2@      &@       @      ?@      9@      ;@      =@      @@      D@     �I@     �C@      A@      E@     �M@      L@     �I@      J@     �D@      F@     @Q@      M@      I@      B@      F@      B@      @@      9@      ,@      @       @        
�
bc1*�	   ��Cv�   �F�?      @@!  �gc��?)���p3tm?2x*QH�x�&b՞
�u�hyO�s�uWy��r�&b՞
�u?*QH�x?o��5sz?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�������:x              �?              �?              �?      �?              @      �?      @       @      @      @        
�
bd1*�	   @Q�t�    ���?      P@!  @1N)�?)B�9@�z?2x&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?�������:x              9@      @      �?      �?      @              �?              �?      �?              @      6@        
�
bo*�	   @�`��   ��[�?     �E@!   �� ¿)޺76��?2����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���;8�clp��N�W�m�uWy��r?hyO�s?���T}?>	� �?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?�������:�              @      *@      @      �?      �?              �?              �?              �?              @              �?              �?      �?      @      @      �?        Sd���"      }-AP	��fo�A*�E

loss��`@

accuracyn4�=
�
wc1*�	   ����   @H��?      r@! ���� @)�[�+i��?2�}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E����#@�d�\D�X=���82���bȬ�0���VlQ.�f�ʜ�7
?>h�'�?�.�?ji6�9�?��bȬ�0?��82?d�\D�X=?���#@?�!�A?�T���C?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�������:�              �?              �?               @      �?      �?       @       @      @      @      @      @      @      @       @       @      @      �?      �?       @       @       @       @       @      �?              �?      �?              �?       @              �?       @               @              �?              �?      �?              �?              �?              �?              �?              �?               @      �?               @              �?      @       @       @      �?      @      @      �?       @              @      @       @       @      @       @       @      @      @       @      @      @      $@      0@      (@      1@      $@      "@      &@      &@      $@      ,@      @      @      �?        
�
wd1*�	   ����   �g��?      A!����ǒ@)֞�cyK@2��/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ��5�L�����]������|�~���MZ��K���u��gr��X$�z��
�}����E'�/��x��i����v��H5�8�t�ہkVl�p�w`f���n��H5�8�t>�i����v>��z!�?�>��ӤP��>X$�z�>.��fc��>�MZ��K�>��|�~�>���?�ګ>����>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?�������:�              �?       @      d@     ��@     ��@     ��@     ��@     z�@     ��@     3�@     h�@    �:�@    ���@     �@    �v�@     +�@     j�@     ��@     һ@     T�@     ��@     m�@     ʴ@     ��@     ��@     `�@     T�@     �@     ��@     ,�@     �@     z�@     �@     ��@     l�@     ��@     0�@     �@     d�@     �@     L�@     8�@     ��@     (�@     ��@     ��@     ��@     ��@     �|@     �{@      z@     Pu@     �t@     �r@     q@     @l@      h@     �h@      k@     �b@     �c@     `b@     �c@      ]@     @W@      Z@     �V@     @T@     @S@     @P@     @P@     �N@      G@     �F@      @@      F@      =@     �C@      ;@      >@      1@      3@      *@      5@      *@      0@      *@      @       @      (@      $@       @      @      $@      (@      @      @       @      �?       @      @      �?      @      �?       @               @       @      �?              �?       @              �?               @              �?      �?              �?              �?              �?              �?              �?              �?              �?              @      @       @      @       @      @      @      @      @      @      @       @       @       @      $@      @      "@      *@      *@      .@      $@      *@      ;@      :@      7@      9@      =@      6@      ;@     �C@      D@     �E@     �D@     @P@     �H@     @Q@     �K@      S@     �R@     �V@     �U@      X@     @_@     �_@     @c@      e@     �d@     `j@     @l@     �h@     �p@     �p@     �t@     �u@     �u@     �z@      }@     `@     �@     ��@     8�@     (�@     ��@     ��@     `�@     0�@     p�@     X�@     ��@     (�@     �@     ̝@     �@     �@     D�@     l�@     ��@     ĩ@     *�@     B�@     ��@     �@     L�@     B�@     �@     ��@     Ž@     \�@     ��@     �@     ��@    ��@     ��@    ���@    �W�@     }�@    ���@    ��@     ,�@     =�@     ��@     �@     ]�@     ��@     ��@     �@      I@        
�
wo*�	   `�f��    �ӟ?     ��@! ���x�)RlVFF�?2��uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.���vV�R9��T7����5�i}1�1��a˲���[��O�ʗ�����Zr[v��pz�w�7�>I��P=�>1��a˲?6�]��?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�������:�              @       @      ,@      ?@      E@     �C@     �S@      O@     @Q@     @R@     �T@      P@      Q@      I@      R@      I@     @P@      H@     �I@     �D@      I@      D@     �C@      B@      A@     �A@      8@      4@      .@      6@      3@      4@      *@      0@      0@      3@      @      "@      $@      @      @      @      @      @      @      @      @      @      �?       @       @              @       @              �?              �?      @      @      �?              �?      �?              �?              �?              �?              �?              �?      �?      �?               @              �?      �?       @      �?              @       @      @      �?       @      @      @      �?      @      �?      @      @      @      @      @      @      @      "@      �?      0@      &@      .@      ,@      &@      .@      0@      .@      >@      8@      :@      =@     �B@     �E@      C@      F@      C@      D@     �K@     �L@      K@      L@     �C@      J@     �M@      N@     �J@     �A@     �H@      ?@     �@@      <@      "@      @      �?        
�
bc1*�	   �ԣv�    G&�?      @@!   �q��?)3��2�l?2p*QH�x�&b՞
�u�hyO�s�&b՞
�u?*QH�x?o��5sz?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�������:p              �?      �?              �?      �?              @      �?      @      @      @      @        
�
bd1*�	   ��u�   `�ȏ?      P@!  ��T�?)��b�{?2�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�������:�              9@      @              �?      @              �?              �?      �?              @      3@      @        
�
bo*�	    �Q��    �N�?     �E@!  �s�}¿)�J��e�?2��Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���P}���h�Tw��Nof�;8�clp?uWy��r?o��5sz?���T}?>	� �?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?�������:�              �?      $@       @      @      �?      �?              �?              �?              �?               @      �?              �?              �?      �?       @      @      �?        ��$      ^�N�	t)�fo�A*�H

loss�]`@

accuracyF�v=
�
wc1*�	   `���   ��&�?      r@! ��ߤ�?)H��?2�}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO��qU���I�
����G����#@�d�\D�X=���%�V6��u�w74���VlQ.��7Kaa+�>h�'��f�ʜ�7
���[�?1��a˲?�S�F !?�[^:��"?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?              �?      �?      �?      �?      �?      @      @      @      @      @      @      @       @      �?      @      @       @      �?      @       @       @       @      �?      �?               @              �?       @       @      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?       @              �?      �?      @      �?       @      �?       @      @      �?      �?       @      �?      �?      @      @       @       @      @      �?       @      @       @       @      @       @      @      @      0@      (@      ,@      ,@      &@       @      *@      &@      "@      &@      @      @        
� 
wd1*� 	   �ق��    C��?      A!�ͬ��L�@)�(-/�J@2��/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>�����ӤP�����z!�?��E'�/��x��i����v�w`f���n�=�.^ol�Fixі�W���x��U�K���7�>u��6
�>T�L<�>��z!�?�>�
�%W�>���m!#�>.��fc��>39W$:��>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?�������:�               @      @     �h@     `�@     ��@     ��@     ٶ@     �@     \�@     ��@     ��@    �z�@     [�@     ;�@     ��@    �I�@     ��@     ��@     �@     ��@     v�@     ��@     �@     Z�@     ��@     ��@     z�@     
�@     �@     ��@     `�@     ��@     ��@     �@     ��@     ��@     T�@     �@     ��@     ��@     ��@     ��@     H�@     ��@     ��@     X�@     Ё@      �@     �~@     �y@     �w@     �x@     �s@     Pt@     Pq@     `o@     �m@     �l@      g@     `e@     �c@      _@     �a@     �Z@     @Z@     @Y@     �X@     @T@     �U@      N@     �Q@     �I@      I@      D@     �H@      B@      @@      D@      7@      8@      1@      8@      3@      5@      (@      &@      (@      "@      @      ,@      @      $@      "@      $@      $@      @      @      @      @      @       @       @      @      @      �?      @      �?      �?       @      @              @       @      �?       @       @      �?              �?      �?      �?               @              �?              �?              �?              �?              �?      �?      �?              �?               @              �?              @      �?      �?       @      �?               @      @              @      @      �?       @       @       @       @      @      $@      "@      @       @      &@      .@      1@      (@      &@      .@      ,@      0@      6@      .@      ;@      8@      =@      D@     �F@     �G@     �B@      H@      H@     @Q@     �P@      S@      R@      Y@      R@      \@     �]@     �`@      b@      f@     �e@     �i@     �k@     @l@     0p@     @s@     u@     `s@     �t@     Pz@     �|@     �~@     H�@     ��@     P�@     ��@     0�@     (�@     8�@     �@     ��@     L�@      �@     ԗ@     ؚ@     D�@     8�@     ��@     Σ@     �@     Z�@     �@     ��@     G�@     ��@     q�@      �@     ��@     f�@     a�@     h�@    �v�@    �d�@     �@    �k�@     =�@     Y�@    ���@    ��@     �@     =�@     ��@    �f�@     ��@     ��@     ��@     �@     q�@     8�@     ��@      E@        
�
wo*�	    ����   `�f�?     ��@! ��ʷ��)W97l�?2��uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��f�ʜ�7
������8K�ߝ�a�Ϭ(�K+�E��Ͼ['�?�;pz�w�7�>I��P=�>��Zr[v�>>�?�s��>�FF�G ?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�������:�              @      @      0@      >@     �G@      C@     @R@     �P@      O@      R@     @T@      R@      P@     �K@      M@      M@     �L@     �J@      I@      G@      G@      F@     �C@      >@     �B@     �A@      8@      1@      5@      9@      5@      ,@      2@      "@      ,@      1@       @      &@      "@      @       @      @      @      @      @      @      @       @      @      @       @      �?               @      @              @       @      �?      �?      @      �?      �?               @              �?               @              �?              �?              �?      �?              �?              �?              �?      �?      �?              �?              @      @      @      @      @      @       @      @      @      "@       @       @      @      �?      @      &@      @      (@       @      $@      *@      &@      "@      ,@      0@      3@      =@      9@      ;@      @@     �@@      F@      F@     �B@     �D@     �E@      K@      M@      J@     �L@      B@     �L@      M@     �L@      M@     �@@      J@      @@      >@      9@      &@      @      �?        
�
bc1*�	    `�v�   `��?      @@!   1���?)kڵa�l?2x*QH�x�&b՞
�u�hyO�s�&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�������:x              �?      �?              �?              �?      �?       @      �?       @      @      "@       @        
�
bd1*�	   ��ou�   ��F�?      P@!  �t�x�?)�dl�r|?2�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k����T}?>	� �?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�������:�              :@      @               @      @              �?              �?      �?              @      0@       @        
�
bo*�	    `P��    #8�?     �E@!   p��¿)�v�}�E�?2��Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L������=���>	� ��E��{��^��m9�H�[��N�W�m?;8�clp?*QH�x?o��5sz?���T}?����=��?���J�\�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�������:�              �?      *@      @       @      �?      �?              �?              �?              �?              �?       @              �?               @       @      @      @      �?        �S�j$      y�6�	��ho�A*�H

lossox`@

accuracy�Jj=
�
wc1*�	   `!*��   @L��?      r@! �h*��?)�&�7�?2�}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L�
����G�a�$��{E��!�A����#@���%>��:�uܬ�@8�+A�F�&�U�4@@�$�ji6�9���.��>h�'��f�ʜ�7
��FF�G �>�?�s����uE���⾮��%��7Kaa+?��VlQ.?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?              �?      �?      �?      �?       @      @      @       @      @      @      @      @      �?       @      @      @               @      @      �?       @       @      �?      �?      �?       @       @      �?              �?              �?      �?              �?              �?               @               @              �?              �?              �?              �?              �?              �?              �?       @      �?       @      �?               @               @       @      �?      �?               @      �?      �?              �?      @       @      �?       @       @      @      @      �?      @       @      @      @      @      @      @      &@      .@      &@      ,@      $@      (@      @      (@      (@      (@       @      @      @        
�
wd1*�	    ��   @-��?      A!�D��@)D�:�YYJ@2��uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ��������|�~���MZ��K���u��gr��R%������39W$:����
�%W����ӤP���7'_��+/>_"s�$1>ہkVl�p>BvŐ�r>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?�������:�              �?       @      @      n@     ��@     �@     X�@     �@     \�@     ��@    �T�@    ��@    ���@     e�@    ���@     �@    ���@     5�@     	�@     :�@     �@     ��@     ��@     W�@     p�@     s�@     ��@     t�@     ��@     $�@     ��@     �@     �@     R�@     P�@     X�@     H�@     x�@     ��@     ��@     �@     ��@     ��@     P�@     0�@     `�@     P�@     0�@     p�@     �{@     `{@     @w@     Pw@     `v@     �s@     �s@     �m@     @l@     �j@     @g@     `d@     @e@     `b@     @b@     �[@      Y@     �X@     �\@     �Q@     �Q@     �L@     �R@      J@      N@      G@      B@      @@     �G@      ?@      >@      A@      0@      7@      6@      ,@      9@      *@      4@      (@      (@      @      &@      *@      @       @      "@       @       @      @       @      @       @      �?      @      @       @      @       @      @      �?       @              �?               @      �?               @              �?              �?              �?      @              @      �?              �?      �?      �?       @              �?      �?               @      @      �?       @      �?      @      @               @      @      @              @      @       @      @      @      &@      $@      @      1@      &@      ,@      &@      (@      1@      8@      $@      4@      7@      :@      ;@      <@      7@      ?@     �E@      B@      D@     �H@     �N@     �M@     �R@      S@     @P@     �W@     �Z@     @_@     �]@     �`@      c@     �d@     �h@     �f@     �i@     �n@     q@     �q@     �r@      u@     w@      z@     ~@     @~@     ��@     X�@     0�@     ��@     ��@     0�@     $�@     0�@     ؒ@     ��@     p�@     ��@     ��@     @�@     p�@     Т@     ��@     ޥ@     ��@     p�@     ��@     ��@     g�@     ?�@     -�@     c�@     Z�@     ޻@     �@    �y�@    ���@    ���@    ���@     �@     M�@     ]�@    ���@     ��@    ���@    �0�@     ��@    ���@     v�@     5�@     k�@     a�@     ~�@     ��@      B@        
�
wo*�	    ͡�   `��?     ��@! �BA��)#���g��?2�	�uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���h���`�8K�ߝ��uE����>�f����>��(���>a�Ϭ(�>��[�?1��a˲?x?�x�?��d�r?�T7��?�vV�R9?��ڋ?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�������:�	              @      @      0@      >@     �F@     �C@     �Q@      N@     �R@      O@     @S@     �R@     �P@     �L@     �O@     �I@     �M@      G@      H@     �H@     �K@     �D@     �C@      :@      A@      D@      9@      .@      2@      7@      5@      2@      1@      1@      .@      ,@      *@      @      @      @       @      @      �?      @      @      "@      �?       @      @      �?      �?      @      @      �?      �?      �?      �?              �?      �?              �?      �?              �?      �?      �?              �?      �?      �?      �?              �?              �?              �?              �?              �?              �?      �?               @              @               @      @      �?       @       @      @      @       @      @      @      @      @      @      @      @       @       @       @      "@      *@      $@      *@      (@      &@      @      ,@      .@      4@      <@      @@      =@      2@     �C@      E@     �D@     �E@     �G@      B@      M@     �L@      H@     �L@     �D@     �L@     �K@     �N@     �L@     �A@      H@     �B@      :@      7@      (@      @      �?        
�
bc1*�	   @�Jw�   ���?      @@!   A�}�?)mA��gl?2p*QH�x�&b՞
�u�hyO�s�&b՞
�u?*QH�x?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�������:p              �?      �?              �?              �?      @      �?       @      @      "@       @        
�
bd1*�	   ��u�   `s��?      P@!  �����?)?9��W6}?2x&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m����T}?>	� �?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�������:x              ;@       @              @              �?              �?      �?               @      .@      $@        
�
bo*�	    8Z��   �6�?     �E@!  @)�ÿ)P6�|�,�?2�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L��>	� �����T}�k�1^�sO�IcD���L�P}���h?ߤ�(g%k?&b՞
�u?*QH�x?o��5sz?����=��?���J�\�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�������:�              �?      @      $@      @      �?               @              �?              �?              �?              �?       @              �?               @       @       @      @      �?        �d�ɜ%      [J%	��ho�A*�K

loss�`@

accuracyHP|=
�
wc1*�	   ��9��   @�O�?      r@! ���4��?)��2Z��?2�}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:���bȬ�0���VlQ.��7Kaa+�I�I�)�(���ڋ��vV�R9��ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>����?f�ʜ�7
?>h�'�?�vV�R9?��ڋ?�.�?+A�F�&?I�I�)�(?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?               @               @               @      @       @      @      @      @      @      @       @       @      @      �?      @      @       @      �?      �?      @      �?       @       @      �?      �?              �?              �?               @              �?      �?       @              �?       @              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?              �?              �?      �?      �?              �?              �?              �?      �?      �?      �?       @              �?               @      @       @              �?      �?      @       @       @      �?      @      �?      @      @      @      @      @      @      1@      (@      (@      0@       @       @      &@      $@      &@      *@      @      @      @        
�!
wd1*�!	    _[��    ӡ?      A!&���э@)8R���I@2��uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L����|�~���MZ��K���u��gr��X$�z��
�}�������m!#���
�%W����ӤP���u��6
��K���7��[#=�؏�������~�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_�6��>?�J���8"uH�������M>28���FP>BvŐ�r>�H5�8�t>�����~>[#=�؏�>�4[_>��>
�}���>X$�z�>.��fc��>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?�������:�              �?      @      $@     �q@     �@     ��@     |�@     M�@     ƺ@     O�@    �{�@     .�@     @�@     ��@     ��@     F�@    ���@     ��@     �@     ��@     ��@     Ƹ@     �@     x�@     B�@     ֱ@     ��@     ��@     ��@     ��@     ��@     F�@     ��@     ʡ@     �@     l�@     <�@     ��@     p�@     Ԕ@     ��@     \�@     ��@     ��@     X�@     0�@     ؅@     ��@     X�@     �~@     �|@     Px@      v@     �s@     @t@     0s@      n@      l@      j@     `h@     �d@     �e@      b@      a@     �]@      [@     �X@     @Y@     @V@     �Q@      K@     �P@     �P@      K@      F@      B@     �H@      F@      9@      @@      8@      8@      5@      5@      :@      1@      @      0@      2@      .@      1@      @      @      "@      @      @      @      @      @       @      @      @      @      @      @       @               @       @      @      @      �?      �?       @              �?       @               @              �?      �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?               @              �?       @      �?      @      �?      @              @      @               @       @      @      @       @       @      @      @       @      @      .@      &@      &@      $@       @       @       @      &@      5@      9@      4@      5@      8@     �B@      8@      ?@      A@      ?@     �A@      G@      J@     @Q@      F@     �L@     �T@      U@     �W@     �U@     @Z@      ^@     �`@     `c@     �e@      g@     �h@     @j@     �k@     @q@     �q@     �s@     0u@     Pw@      z@     �}@     �~@     (�@     `�@     ��@     �@     8�@      �@     |�@     `�@     ؒ@     \�@     ��@     ��@     x�@     Ğ@     ��@     �@     ƣ@     ֥@     L�@     ��@     �@     (�@     |�@     X�@     Z�@     p�@     ��@     ��@     �@     v�@    ���@    ���@     ��@    ���@     A�@     6�@     ��@     6�@     ��@    ���@    ���@    �K�@     ��@     Q�@     J�@     ��@     �@     Ѓ@      C@        
�
wo*�	   �����   �np�?     ��@!  c�,��)�������?2�	�uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !��.����ڋ��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���>�?�s���O�ʗ����h���`�8K�ߝ�
�/eq
Ⱦ����ž;�"�q�>['�?��>�ѩ�-�>���%�>1��a˲?6�]��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�������:�	              @       @      ,@      A@     �C@      I@     �M@     �O@     �Q@     �P@     �R@     �Q@      Q@     �N@      P@      I@      K@     �G@      E@      M@      I@     �E@     �B@      @@      B@      A@      ;@      (@      =@      3@      0@      1@      &@      7@      &@      1@      "@      &@       @      @      @      @      @      @      @      @       @      @       @      @      @              �?      �?               @       @               @              �?              �?               @      �?      �?              �?              �?              �?              �?              �?              �?              �?              @      �?      �?              �?              @      �?      �?       @       @      @      @      @      @      �?      @       @      @      @      @      @      @      @      .@       @      *@      ,@      &@      &@      $@      *@      4@      :@      7@      >@      7@      4@      B@     �C@      H@      F@      H@      A@     �J@      N@     �J@     �K@     �E@      L@      N@      L@      L@      E@     �D@     �C@      9@      8@      &@      @      �?        
�
bc1*�	   �j�w�   `�$�?      @@!   #,��?)=�t��l?2p*QH�x�&b՞
�u�hyO�s�&b՞
�u?*QH�x?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�������:p              �?      �?              �?               @      �?       @      @      @      @      @        
�
bd1*�	   ���u�   ���?      P@!   ���?)2�T~?2�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m����T}?>	� �?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�������:�              5@      @       @              @              �?              �?      �?               @      (@      *@        
�
bo*�	   �km��   �b��?     �E@!  �ӦĿ)�x��!�?2�^�S�����Rc�ݒ����&���#�h/���7c_XY��eiS�m��-Ա�L��>	� �����T}�U�4@@�$��[^:��"�5Ucv0ed?Tw��Nof?hyO�s?&b՞
�u?*QH�x?o��5sz?����=��?���J�\�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?�������:�              �?      "@      @      @               @              �?              �?              �?              �?      �?      �?              �?               @      �?       @      @      @      �?        {��\%      Y���	}[�io�A*�J

loss:a@

accuracyl	y=
�
wc1*�	   �;H��   ���?      r@!  �*��?)�������?2�}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@���%>��:�uܬ�@8��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�U�4@@�$��[^:��"��vV�R9��T7����5�i}1�pz�w�7�>I��P=�>1��a˲?6�]��?�7Kaa+?��VlQ.?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�T���C?a�$��{E?
����G?�qU���I?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?               @               @               @       @      @      @      @      @      @      @      @      @      @       @      @      @      �?      �?       @       @      @      �?       @              �?              �?       @      �?      �?              �?       @               @              �?              �?               @       @       @              �?              �?              �?      �?              �?              �?              �?              �?              �?      �?              �?               @              �?              �?      @       @       @      �?               @       @      �?       @       @              @              @      @      @      @      @      @      "@      ,@      $@      *@      *@      "@       @      &@      "@      &@      *@      @      @       @        
� 
wd1*� 	   @6���    p��?      A!Y{G�yP�@)������I@2��uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%��������ӤP�����z!�?��K���7��[#=�؏��ہkVl�p�w`f���n�%���>��-�z�!>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>E'�/��x>f^��`{>[#=�؏�>K���7�>T�L<�>��z!�?�>��ӤP��>���m!#�>�4[_>��>
�}���>X$�z�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>����>豪}0ڰ>��n����>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?�������:�               @      @      (@     �s@     |�@     ��@     ��@     t�@     /�@     ��@     ��@    �b�@     g�@     ��@    ��@     u�@     ��@     �@     ��@     ��@     �@     ��@     *�@     ��@     �@     �@     �@     (�@     ��@     ��@     ��@     ��@     ֢@     ^�@     �@     �@     `�@     ̘@     �@     ��@     ��@     �@     P�@     h�@      �@     ��@     ��@     P�@      �@     �~@     P|@      z@     �w@     `w@     0s@     �q@     @p@      n@     �f@      h@     �f@     �c@     `d@      a@     �Z@     �[@     �^@     �V@      T@      U@     �V@      R@      M@     �M@      G@      G@      F@     �A@      =@      3@      :@      5@      6@      4@      7@      2@      (@      0@      &@      &@      (@      "@      @      "@       @      @      @       @      @      @       @      @      @      �?       @       @       @       @              �?      �?      �?       @       @              �?      �?               @              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?               @      �?              �?       @              @      �?      @       @      @      @      @      "@      @      @      $@      @      $@      0@      &@      &@      "@      "@      @      $@      1@      3@      8@      4@      <@      2@     �@@     �D@     �A@      =@     �E@      Q@      M@     �J@      O@     �S@     �T@      W@     �R@     �^@     �[@     �a@     �_@     �d@     �j@     �h@     �j@     `m@     �p@     Pr@     `r@     �u@     0v@     �|@      |@     H�@     ��@      �@     ��@     `�@     `�@     @�@     L�@     ��@     T�@     ��@     t�@     ��@     l�@     �@     ��@     �@     ��@     B�@     ب@     ��@      �@     ��@     Y�@     ��@     p�@     p�@     ��@     ֻ@     "�@    �_�@    ���@    ��@    ���@     ��@     =�@     ��@    ���@     �@    �Q�@    �P�@    �M�@     �@    �x�@     ��@     ��@     ��@     �@     ��@      F@        
�
wo*�	   �X5��    �?     ��@! @o�u��)_��8� �?2�	�uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.��>h�'��f�ʜ�7
������6�]����FF�G �>�?�s�����>M|Kվ��~]�[ӾE��a�W�>�ѩ�-�>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�	              @      &@      ,@      A@      C@     �H@      O@      L@     �Q@      Q@      R@     @R@     @Q@      L@     �P@      H@      K@     �H@      D@      N@     �F@      G@     �A@     �B@      B@      >@      @@      0@      9@      1@      5@      ,@      (@      4@      @      4@      &@       @      @      @      @      @       @      @      �?      @      @      "@      @       @      �?      �?       @       @              �?      �?       @               @      �?              �?      �?      �?              �?              �?              �?              �?              �?      �?              �?       @       @              �?               @      �?               @              @      �?      @              �?       @      @      @      �?      @      @      @      @      @      @      @      @      @      @       @      $@      (@      $@      ,@      *@      "@      0@      $@      ;@      9@      2@      ?@      5@      9@      @@      D@     �G@     �F@     �H@     �A@     �H@     @P@      J@      I@     �I@      K@     �L@      O@     �J@     �C@     �F@     �B@      :@      9@       @      @        
�
bc1*�	   @X�w�   `��?      @@!   ����?)4R��ڛm?2p*QH�x�&b՞
�u�hyO�s�&b՞
�u?*QH�x?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�������:p              �?      �?              �?              �?       @      �?      @      "@      @      @        
�
bd1*�	   @�5v�   �ؖ�?      P@!  @��T�?)U�� :?2�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�o��5sz?���T}?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?�������:�              7@      @       @              @              �?              �?              �?       @      "@      .@      �?        
�
bo*�	   @⇕�   ��˕?     �E@!   .��Ŀ)=w�t)�?2��"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����T}�o��5sz��!�A?�T���C?�m9�H�[?E��{��^?uWy��r?hyO�s?&b՞
�u?*QH�x?����=��?���J�\�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?�������:�              �?              $@      @      @              �?      �?              �?              �?              �?              �?      �?      �?              �?               @      �?       @      @      @      �?        6�.�\%      Y���	@2Kjo�A*�J

loss�	a@

accuracyCk=
�
wc1*�	   @KU��   `�՜?      r@!  b�ߔ�?)�
~P3�?2�}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�})�l a�>pz�w�7�>��d�r?�5�i}1?�[^:��"?U�4@@�$?��bȬ�0?��82?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?               @               @      �?       @      "@       @      @      @      @      @      @      @              @       @      @      @      �?      @       @      @      �?      �?      �?               @      �?       @      �?              @       @              �?      �?      �?       @       @       @      �?              �?      �?              �?              �?              �?              �?              �?              �?      �?               @              �?              �?               @      �?      �?       @      �?               @      �?       @      �?       @       @              @       @       @      @      @      @      @      @      @      "@      1@      "@      .@       @      $@      "@      &@      "@      "@      *@      @      @       @        
� 
wd1*� 	   @���   @>'�?      A!7%�wp�@)m�J�I@2��uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ������;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>����
�%W����ӤP�����z!�?��K���7��[#=�؏��BvŐ�r�ہkVl�p��
L�v�Q�28���FP�ہkVl�p>BvŐ�r>[#=�؏�>K���7�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>����>豪}0ڰ>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?�������:�              @      @      4@      t@     H�@     *�@     �@     ��@     c�@     ƾ@     ��@    ���@     ��@     
�@     �@    ���@    ��@     �@     Ҿ@     ��@     �@     <�@     P�@     ]�@     5�@     M�@     �@     |�@     b�@     &�@     ��@     
�@     �@     N�@     B�@     ��@     H�@     ��@     T�@     Ĕ@     ��@      �@     �@     ��@     ��@     h�@     ȅ@     h�@     ��@     8�@     �|@     `z@      v@     �t@     �s@     �q@     pp@     `l@     `j@      i@     �f@      h@     @a@      b@     �`@     @Z@     �]@      V@     @W@     @Q@     @T@     @R@      H@     �P@     �C@      E@     �F@      =@      8@      6@      8@      8@      7@      2@      5@      3@      0@      4@      $@      "@       @      &@      @      @      @      @      $@      @      @      @      �?      "@      @      �?      @      �?              @      @      �?      �?              �?      �?       @              �?      �?              �?      �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?      �?               @       @              �?               @              @       @       @       @       @      @      @      @      @      @      @      @      @      @      (@      @      &@      &@      "@      "@      2@      .@      1@      0@      (@      5@      3@      ?@      7@      8@      F@     �C@      <@     �G@      N@     �K@      P@      O@     �T@     @U@     �S@     �Z@      Z@      ^@     ``@     �b@     �e@     �g@     �k@     �f@     @l@     Pr@     �r@     0s@     �t@     �x@     �z@     0}@     ��@     (�@     �@     �@     ��@     0�@     �@     ��@     x�@     �@     ̓@     �@     ��@     ��@     L�@     8�@     �@     ԣ@     �@     :�@     6�@     H�@     P�@     ��@     ��@     V�@     p�@     ��@     Ż@     !�@    ���@     b�@     �@    �<�@     ��@    ��@    ��@     ��@     �@     (�@    ��@    ��@    ���@    �+�@     ��@     H�@     F�@     ��@     �@      L@        
�
wo*�	   ��q��   �r��?     ��@! �pJ��)��Q%�?2�	`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&��[^:��"��S�F !��5�i}1���d�r�x?�x��>h�'��f�ʜ�7
�6�]���1��a˲�})�l a��ߊ4F��h���`�
�/eq
Ⱦ����ž�
�%W�>���m!#�>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�	              �?      @      *@      *@      B@      D@      G@     �L@     �M@      R@      Q@     �Q@      S@      Q@     �J@     @P@      J@     �K@      G@      C@      M@      G@     �D@      E@      @@      B@     �B@      >@      3@      5@      5@      3@      *@      (@      *@      (@      0@      &@      &@      @      @       @      @      @      @      @      �?       @      @      @      @      �?       @      @              �?       @              �?              �?      �?      �?      �?              �?              �?      �?              �?              �?               @              �?              �?              �?       @      �?      �?              �?       @      @      �?      @              �?       @      �?      @       @      �?      �?       @       @      @       @      @      @      @      @      @      @      @      $@      @      .@      "@      ,@      .@      @      ,@      &@      ;@      6@      :@      8@      :@      7@      >@     �E@     �H@      C@     �G@      D@      J@      P@      I@     �H@      L@      J@     �M@     �N@     �H@      G@      D@      E@      6@      :@      @      @        
�
bc1*�	   �^x�   ���?      @@!   0?d�?)��m�"o?2xo��5sz�*QH�x�&b՞
�u�hyO�s�&b՞
�u?*QH�x?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�������:x              �?              �?              �?              �?       @      �?      @      $@      @      "@        
�
bd1*�	   ��lv�   �H�?      P@!  ��D��?)m�v�G�?2�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�o��5sz?���T}?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?�������:�              8@      @       @              @              �?              �?              �?      �?      @      .@      @        
�
bo*�	   ����   �N��?     �E@!  ���cſ)�%��rG�?2��"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����T}�o��5sz��qU���I?IcD���L?�lDZrS?<DKc��T?;8�clp?uWy��r?&b՞
�u?*QH�x?����=��?���J�\�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?�������:�              �?       @      "@       @       @              �?      �?              �?              �?              �?               @              �?              �?               @      �?      �?      �?      @      @        ��Z�L%      E���	{��jo�A*�J

loss'�`@

accuracyjMs=
�
wc1*�	   @0a��   `(��?      r@! ��7p�?)�ྙ,	�?2�}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:���bȬ�0���VlQ.��7Kaa+�I�I�)�(��vV�R9��T7���x?�x�?��d�r?�[^:��"?U�4@@�$?�u�w74?��%�V6?��%>��:?d�\D�X=?���#@?�!�A?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?              �?       @      �?      �?       @      "@      @      @      @      @      @      @       @              @       @      @       @               @      @      @      �?      �?      �?      �?      @              �?      @      �?      �?       @      �?              �?      @       @       @      �?              �?              �?              �?              �?              �?               @              �?      �?      �?              �?      �?              �?       @      �?      �?      �?       @      �?      �?              �?       @      @      �?      @              @      @      @      @      @      @      @      ,@      *@       @      ,@      "@      (@      @      *@       @       @      *@      @      @       @        
�!
wd1*�!	   `�z��   @b�?      A!j$�k~-�@)������I@2�`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc����4[_>������m!#���
�%W��K���7��[#=�؏�������~��H5�8�t>�i����v>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>.��fc��>39W$:��>R%�����>�u��gr�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�������:�              �?       @      @      9@     �t@     ԛ@     j�@     �@     ŷ@     ��@     ��@    ��@    ���@    ���@     �@    �)�@     v�@    �!�@     �@     Ӿ@     �@     ��@     @�@     8�@     H�@     �@     ��@     ,�@      �@     ��@     �@     h�@     �@     B�@     ,�@     j�@     `�@     ��@     ,�@     ��@     \�@     �@     p�@     H�@     ؋@     �@     P�@     ��@     �@     0@     �@     @z@     �y@     pu@      w@     0t@      r@      n@     �m@     �n@     @i@     �i@     @e@     @b@     �^@     �a@      ]@      [@      Z@     �[@     �P@      Q@      M@      O@     �G@      H@      E@     �F@     �B@      7@      7@      9@      7@      9@      1@      *@      (@      .@      2@      ,@      "@      1@      @      �?      @       @       @      @      @      "@      @      @      @      @      �?      @       @      �?              @      �?       @      �?       @      �?      �?      �?      �?               @              �?               @      �?              �?      �?              �?              �?      �?      �?              �?      �?              �?      �?      �?              @      @      �?       @              �?       @      �?       @      @      @      @      @       @      �?      �?      @      @      (@      @      @      @      "@      @      &@      *@       @      "@      "@      1@      .@      (@      3@      ;@      :@      :@      E@      7@      =@      <@     �@@     �C@      O@      K@      J@      R@     @T@     �R@      W@     @W@      ]@      [@     �a@      c@     �f@     `f@     `j@     �i@     �j@     �q@     Pr@     �t@     �u@     �w@     �z@     P|@     @�@     h�@     ��@     ��@     ��@     �@     8�@     H�@     ��@     p�@      �@     Ж@     �@     `�@     ؝@     6�@     ¢@     ޣ@     8�@     ��@     l�@     N�@     �@     �@     <�@     0�@     M�@     ǹ@     ��@     �@    �o�@     ��@     ��@     (�@    ���@    ���@    ��@     ��@     �@    �
�@     �@     ��@     ��@    �)�@     ��@     ]�@     n�@     ��@      �@      S@      �?        
�
wo*�	    ����   ��Z�?     ��@! ��s.�)\b���`�?2�	`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+��[^:��"��S�F !��5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[��O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`��?�ګ>����>
�/eq
�>;�"�q�>a�Ϭ(�>8K�ߝ�>����?f�ʜ�7
?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�	              @      @      *@      2@      A@     �D@      G@      M@     �L@     �R@     �Q@      Q@     �S@     �N@      L@      N@      J@     �O@     �G@      B@     �K@      G@      F@      A@     �D@      ?@      B@      ?@      2@      6@      3@      6@      (@      ,@      "@      ,@      0@      $@      "@      @       @      @      @      �?      @      @      @      @      @       @       @      @      @       @      �?               @              �?      �?              �?              �?              �?              �?              �?      �?      �?      �?              �?              �?              �?              �?              �?      �?       @      �?       @      �?       @      @       @      @      @      �?      @       @              @      @      @      @      �?      @      @      @      @      @      &@      @      &@      *@      3@       @      (@      "@      &@      (@      6@      2@      <@      <@      6@      :@     �B@      D@      F@      E@     �C@     �F@      K@     �O@      K@      F@      L@     �H@     �N@     @P@      H@      G@     �D@      D@      7@      :@      @      @        
�
bc1*�	    	Hx�   `Rً?      @@!  �-g�?)��!�~p?2po��5sz�*QH�x�&b՞
�u�hyO�s�&b՞
�u?*QH�x?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�������:p              �?              �?              �?               @       @       @      "@      @      "@        
�
bd1*�	   `a�v�   �߮�?      P@!  @�|�?)�Q6��?2�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�o��5sz?���T}?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?�������:�              9@      @      �?      �?      @              �?              �?              �?              @      *@      $@        
�
bo*�	    =ʗ�   �y�?     �E@!   �o$ƿ)�Z�}�?2�}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L��o��5sz�*QH�x�>�?�s���O�ʗ�����bB�SY?�m9�H�[?ߤ�(g%k?�N�W�m?;8�clp?&b՞
�u?*QH�x?����=��?���J�\�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�������:�              �?              @      @      @              �?      �?      �?              �?              �?              �?              �?      �?              �?              �?              �?       @      �?      �?      @      @      �?        M���<%      X�	O�ko�A*�J

lossE�`@

accuracy�{r=
�
wc1*�	   �l��    $Ü?      r@!  ��
��?)�<�X�?2�}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L�
����G�a�$��{E��T���C��!�A����#@���%>��:�uܬ�@8��u�w74���82���ڋ��vV�R9���d�r?�5�i}1?ji6�9�?�S�F !?+A�F�&?I�I�)�(?��VlQ.?��bȬ�0?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?�T���C?a�$��{E?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?              �?       @      �?      �?      @      @      @       @       @       @      @      @              @       @      @       @              @       @      @      �?       @      �?      @       @              �?       @      �?      @      �?              �?       @      �?      @              �?              �?              �?              �?              �?              �?              �?              �?               @      �?               @               @      �?              �?              @      �?              �?      �?       @       @       @              @       @       @      @      @      @      @      @      @      (@      .@      @      (@      "@      *@       @      (@       @       @      *@      @      @       @        
� 
wd1*� 	    �Т�   ��Ǣ?      A!�<M�y�@)c�-k�J@2�`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L����|�~���MZ��K���u��gr��R%������39W$:���.��fc����
�%W����ӤP���w`f���n�=�.^ol�ڿ�ɓ�i�w&���qa>�����0c>cR�k�e>[#=�؏�>K���7�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�������:�              �?      @      @      ;@     �u@     4�@     ��@     �@     �@     һ@     ��@    ��@    ���@    ���@     !�@    ��@     z�@     �@     �@     ۾@     Ѽ@     �@     V�@     ��@     �@     _�@     ��@     ٰ@     �@     ��@     l�@     <�@     8�@     �@     8�@     �@     ��@     ؚ@     ��@     �@     ̓@     l�@     8�@     p�@     ��@     �@     ��@     ȅ@     (�@      �@     �~@     �y@     �y@     �w@     �u@     pt@     @s@     Pr@      o@     `h@     �h@     `f@     `f@      b@     `a@     ``@      X@     �]@     �Y@     @U@     �P@      P@     @P@     �K@     �K@      E@      F@      G@      D@      6@      4@      @@      7@      5@      :@      6@      2@      &@      3@      *@      ,@       @      @      "@      @      "@      @      @      @      @      "@       @      @      @       @      @      @              �?      @      @       @       @              �?              �?              @               @              �?              �?      �?              �?      �?              �?              �?              �?      �?              �?              �?               @      �?              �?      �?       @               @      @       @      @      �?      @       @      @       @       @      @      @      @      @      "@      "@      @      &@      1@      ,@      (@      (@      (@      &@      0@      4@      7@      9@      A@      ;@      B@      D@      @@     �@@     �C@      M@     �I@      Q@      O@     @U@     �P@     �T@      V@     �[@     �`@     ``@      b@      d@     �f@     �h@     @k@     �m@     �n@     Pr@     Pt@     0w@     �u@     �x@     �}@     0�@     (�@     ��@     ��@      �@     ȉ@     p�@     ��@     ��@     �@     ��@     ��@     Ԙ@     ��@     ̞@     �@     ^�@     ��@     L�@     ¨@     ֪@     ح@     D�@     ��@      �@     u�@     ��@     ��@     Y�@     2�@     T�@    ���@     ��@     :�@     ��@     ��@     ��@    ���@    ���@    �$�@    �%�@    ���@     �@    �1�@     ��@     ��@     ��@     <�@     8�@     �Z@       @        
�
wo*�	    ��   �@(�?     ��@! `�~�)o�9���?2�	`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"�ji6�9���.����ڋ��vV�R9��T7���x?�x��>h�'�������6�]����FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7�������>
�/eq
�>I��P=�>��Zr[v�>O�ʗ��>1��a˲?6�]��?��d�r?�5�i}1?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�	              @      @      (@      5@     �A@     �D@     �F@      N@      L@     @Q@     @R@     @R@     �R@      O@     �I@      N@      N@     �N@      F@      A@      L@     �G@     �F@      A@     �B@      <@     �B@      A@      1@      7@      6@      .@      1@      ,@       @      (@      3@      $@      "@      $@      @      @      @      @       @      @       @      @      @       @       @      @       @      @      �?      �?      �?              �?              �?       @       @      �?              �?              �?              �?              �?              �?              �?              �?      �?              �?               @               @      �?      @      @       @      �?      @      @      @      �?      �?               @      @      @      @      @      @      @      $@      @      @       @      &@      "@      "@      @      6@      $@      $@      @      (@      *@      5@      3@      7@      <@      ;@      4@      F@     �C@      E@      E@      C@      D@      O@     �L@     �J@     �H@     �J@     �J@      O@     �O@      I@      F@     �E@     �C@      6@      :@       @      @        
�
bc1*�	   ��yx�    �Ɍ?      @@!   ���?)ǪC���q?2xo��5sz�*QH�x�&b՞
�u�hyO�s�&b՞
�u?*QH�x?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?�������:x              �?              �?              �?               @      �?      �?      @       @      @      @        
�
bd1*�	   @��v�   �lJ�?      P@!   b<�?)�?R��?2�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m����T}?>	� �?���J�\�?-Ա�L�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�������:�              9@      @      �?      @       @              �?              �?              �?      @      (@      &@      �?        
�
bo*�	   �    �Q�?     �E@!  �{@�ƿ)~
�4�Ɉ?2�}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L��o��5sz�*QH�x�k�1^�sO�IcD���L��l�P�`?���%��b?Tw��Nof?P}���h?&b՞
�u?*QH�x?����=��?���J�\�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�������:�              �?              "@      @      @               @              �?              �?              �?              �?               @              �?              �?              �?       @      �?      �?      @       @       @        f$��$      \��(	̙wlo�A*�I

loss�`@

accuracyF�v=
�
wc1*�	   ��u��   �Nڜ?      r@!  �sp�?)�g
b�1�?2�}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:���%�V6��u�w74��.����ڋ�pz�w�7��})�l a��[^:��"?U�4@@�$?+A�F�&?I�I�)�(?��VlQ.?��bȬ�0?��%�V6?uܬ�@8?��%>��:?d�\D�X=?�!�A?�T���C?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?      �?               @      �?      �?      @      @      @       @      @      @      @      @              �?      @       @      @       @               @       @      @       @       @       @      �?      �?       @              @      @      �?       @               @      �?              @       @      �?              �?              �?              �?              �?              �?              �?              �?      �?      �?               @              �?      �?              �?               @      @              �?               @       @       @      �?      @      �?       @      @      @      @      @      @      @      $@      0@      "@      &@       @      (@      $@      $@      $@       @      *@      @      @       @        
� 
wd1*� 	   ��!��   ��3�?      A!jn��A;�@)�X��7TJ@2�`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ����?�ګ�;9��R���5�L�����]�����MZ��K���u��gr��39W$:���.��fc���X$�z��
�}����u��6
��K���7��4�e|�Z#���-�z�!���x��U>Fixі�W>:�AC)8g>ڿ�ɓ�i>�i����v>E'�/��x>���m!#�>�4[_>��>X$�z�>.��fc��>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�������:�              �?      @      @      <@     �v@     ��@     ܥ@     �@     �@     Ȼ@     ��@    ��@     ��@     ��@     �@     �@    �\�@    ���@     �@     S�@     ��@     �@     7�@     ʶ@     %�@     S�@     ]�@     ��@     ��@     ��@     T�@     §@     F�@     ��@     J�@     \�@     ��@     ��@     �@     (�@     ��@     4�@     p�@     0�@     0�@     ��@     X�@     0�@      �@     ��@      }@     �z@     �y@     �w@     �v@      v@     �r@     �n@      i@      k@     �i@      g@      d@      b@     �]@     @`@     @\@     �]@     �Y@     �V@     @S@      N@     �G@     �O@     �I@      J@      @@      I@     �B@      2@      5@      >@      8@      5@      6@      8@      (@      5@      5@      "@      ,@      $@      "@      @      @      @      @      @      @      @      "@       @      @      @      �?       @      @      @      @      �?      @              �?              �?               @              �?              �?              �?              �?              �?              �?              �?              �?               @               @              �?      �?               @      �?      @       @      �?      �?      @      �?      @       @       @      @      @       @      @      @      @       @      *@      *@      @      &@      (@      "@      .@      3@      6@      2@      $@      7@      1@      7@      ;@      ?@      @@     �E@      C@      G@     �P@      N@     @R@     @P@     �U@     @Q@     �X@     �V@     �Y@     �V@      d@     @b@     @f@      f@     �j@     �i@     �k@     �p@     �r@     Ps@      u@     �u@     pz@     �{@     p~@     ��@     ��@     `�@     ��@     �@     ��@     �@     ��@     T�@     t�@     ��@     (�@      �@     t�@     �@     &�@     ԣ@     f�@     ��@     ��@     `�@     ]�@     ��@     D�@     %�@     ��@     o�@     �@     "�@     =�@     ��@     ��@     �@     ��@     ��@    ���@    ���@     ��@     I�@    �9�@    ��@     9�@     _�@     ��@     X�@     �@     @�@     ؋@     �c@      @        
�
wo*�	    �>��   `뷝?     ��@! �����)�bzK�?2�	`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9����ڋ��vV�R9��T7���x?�x��>h�'��f�ʜ�7
������1��a˲���[��>�?�s���O�ʗ�����|�~�>���]���>����>豪}0ڰ>I��P=�>��Zr[v�>f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�	              @       @      "@      9@      ?@      F@     �E@      N@      N@      Q@     �R@     �Q@      S@     �M@      K@      M@      M@      O@      F@      C@     �J@      K@     �E@      ?@     �@@      >@     �C@      8@      <@      .@      5@      2@      2@      *@      $@      ,@      0@      "@      "@      &@      @      @      @       @      @       @      @      �?      @      �?      �?       @              @       @      �?      �?              �?      �?      �?               @      �?              �?      @      �?              �?              �?              �?              �?              �?               @               @       @               @      @      �?      �?              �?       @      @      @      �?      @      @      @       @      @      �?       @       @      @      @      @      @      @       @       @      (@      @      *@      $@      $@      $@      (@       @      2@      5@      2@      9@      6@      <@      8@     �A@      F@      F@     �C@      C@      E@     �M@      M@      K@     �I@     �I@     �J@      P@      N@      I@     �F@      F@     �B@      9@      7@      $@      @        
�
bc1*�	   �/�x�   �ݍ?      @@!   	���?)�{��a7s?2xo��5sz�*QH�x�&b՞
�u�hyO�s�&b՞
�u?*QH�x?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?�������:x              �?              �?              �?              �?       @      �?       @      $@      @      $@        
�
bd1*�	   ���v�   @�?      P@!  �O��?)�I6����?2�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m����T}?>	� �?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�������:�              9@      @      �?      @      �?              �?              �?              �?       @       @      ,@      @        
�
bo*�	   ���   �*.�?     �E@!   �r�ǿ)ߤ���+�?2���<�A���}Y�4j��^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��o��5sz�*QH�x�E��{��^��m9�H�[��l�P�`?���%��b?5Ucv0ed?&b՞
�u?*QH�x?����=��?���J�\�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�������:�              �?              (@      @      @               @      �?              �?              �?              �?       @              �?              �?              �?      �?       @      �?              @      @      �?        ^@"Sj%      b��	.mo�A*�J

loss�"`@

accuracy��q=
�
wc1*�	   ��~��   �J��?      r@!  M2���?)�`}�r�?2�}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���VlQ.��7Kaa+���ڋ?�.�?��82?�u�w74?��%�V6?���#@?�!�A?�T���C?�qU���I?IcD���L?k�1^�sO?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?      �?              �?       @      �?      @      @      @      �?      @      @      @      @              �?      @      �?      @      @               @       @      �?      @              @       @              @      �?      @       @               @               @      �?      �?      �?              �?              �?      @       @              �?              �?              @      �?               @       @              �?      �?              �?       @       @      �?      �?              @      @      �?              �?      @      @      @      @      @      @      @      @      $@      0@       @      &@      $@      "@      (@      &@      $@       @      *@      @      @       @        
� 
wd1*� 	    
o��   `���?      A!�4$bP�@)���@[�J@2�`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K��R%������39W$:���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<���H5�8�t�BvŐ�r�ڿ�ɓ�i�:�AC)8g�=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�
�%W�>���m!#�>�4[_>��>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�������:�               @      @      $@      <@      w@     �@     �@     6�@     ��@     Ի@     �@    ���@     ��@     ��@    ��@     ��@     <�@    ���@     ƿ@     g�@     ��@     ��@     �@     ��@     $�@     F�@     =�@     ��@     ��@     ܫ@     ��@      �@     .�@     ΢@     ��@     �@     ��@     Ě@     �@     P�@     (�@     d�@     ��@     @�@     0�@     ��@     ؇@     ��@     P�@     ��@     p~@     z@     �y@     �x@     `w@     `s@     @q@     pp@     @n@     �m@     �h@     `f@     `e@      a@     �_@     �\@     @Y@     �Y@      Z@      S@     @R@     �O@     �H@     �G@     �J@      H@      F@      D@      E@      8@      8@      1@      2@      2@      5@      7@      1@      $@      0@      $@      (@       @      @      (@      @      &@      @      @       @      @      @      @      @      @      @      @              @       @      @       @      �?               @       @       @      �?      �?              �?              �?              �?              �?               @              �?              �?              �?              �?              �?       @              �?      �?      @       @      �?       @      �?      �?      @      �?      �?      @       @       @      @      @      @      @      @      @      @      $@      $@      @      &@      .@      @      @      (@      1@      5@      .@      5@      5@      .@      8@      8@      :@      9@      C@      C@      H@      M@      J@     @R@     �P@     �R@     �R@     �X@     @W@     �Y@     @[@     @^@     �b@     @c@      f@      i@      m@     @l@     �p@     0q@     �r@     �u@     `v@      z@     �}@     ��@      �@     P�@     ��@      �@      �@     ؊@     ��@     А@     <�@     ��@     ��@     ��@     ̛@     ��@     ��@     2�@     ��@     ��@     ��@     �@     ��@     ��@     ı@     [�@     3�@     ��@     4�@     �@     ��@    �G�@    ���@    ���@    ���@     ��@    ���@    ���@    ���@     ��@    �G�@    �/�@    �u�@     p�@     ��@     9�@     1�@     �@     ��@     8�@      l@      (@        
�
wo*�	    ]���   @jM�?     ��@!  Nfm��)��β��?2�	`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���5�i}1���d�r�x?�x��>h�'��1��a˲���[���FF�G ���Zr[v��I��P=��pz�w�7���ߊ4F��h���`�8K�ߝ�a�Ϭ(��uE���⾮��%ᾙѩ�-�>���%�>})�l a�>pz�w�7�>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�	              @      &@      "@      8@     �B@     �D@     �D@      P@     �N@     @P@     �S@     �Q@     �R@      K@      M@      O@      L@      N@      F@      C@     �K@      L@      B@      >@     �@@     �A@      @@      7@      :@      6@      3@      2@      3@      (@      *@      &@      2@       @      @       @      @       @      @      @      @      @       @       @      @      @       @      @              �?      @       @      @      �?       @       @       @               @               @              �?      �?               @      �?              �?              �?              �?              �?              �?              �?      �?              �?      �?               @      �?      @       @      �?      �?              �?              �?      @       @               @      @      @      @      @      �?              �?      @      @      @      @      @      @      @      $@      @      *@      $@      .@      "@      $@      *@      *@      4@      5@      :@      8@      6@      >@      @@      D@      G@      A@     �D@     �F@      K@     �P@      H@     �K@     �G@     �J@     �P@     �N@      I@     �E@     �E@      A@      <@      9@      "@      @        
�
bc1*�	   ���x�   ��{�?      @@!  �����?)j�r��t?2xo��5sz�*QH�x�&b՞
�u�hyO�s�*QH�x?o��5sz?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�������:x              �?              �?              �?               @       @              @      $@      "@      �?        
�
bd1*�	   @�w�   �)��?      P@!  ��|��?)8��?2x*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp�>	� �?����=��?-Ա�L�?eiS�m�?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�������:x              9@      @      �?      @              �?              �?               @      @      *@      @        
�
bo*�	    �0��    ��?     �E@!  �f�ȿ)XA��b��?2���<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��o��5sz�*QH�x�Tw��Nof�5Ucv0ed�ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?5Ucv0ed?Tw��Nof?&b՞
�u?*QH�x?����=��?���J�\�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�������:�              �?              @       @      @      �?      �?      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              @      �?              @      @      �?        $���&      <���	��mo�A*�M

loss��_@

accuracy�7x=
�
wc1*�	   @���   �M,�?      r@!  cB��?)i؛��ä?2�}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A�d�\D�X=���%>��:�uܬ�@8���%�V6���82���bȬ�0��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$���bȬ�0?��82?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?      �?              �?       @      �?      @      @      @      �?      @      @      @      @      �?      �?      @       @       @      @              �?       @      �?      @       @       @      �?       @      �?       @       @      �?      �?      �?              �?      @      �?      �?              �?      �?      �?              �?              �?      �?       @              �?               @              �?      �?              @      �?      �?              �?              �?               @       @      �?      �?      �?               @       @      �?      �?       @      @      �?      @      "@      @      @      @      @      &@      (@       @      *@      &@      "@       @      *@      &@      "@      *@      @      @      @        
�!
wd1*�!	   �$���    3�?      A!��F���@)���1K@2�`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP���T�L<��u��6
��K���7��[#=�؏�������~�����W_>�p
T~�;�u��6
�>T�L<�>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>39W$:��>R%�����>�u��gr�>��|�~�>���]���>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�������:�               @      @      *@      9@     �w@     �@     �@     N�@     �@     ػ@     �@    ���@     ��@     d�@     ��@    ���@    �/�@     ��@     ��@     �@     Q�@     ��@     ��@     ��@     ��@     �@     l�@     ��@      �@     ��@     ��@     ��@     ¤@     �@     ڡ@     ��@     �@     X�@     ��@     (�@     t�@     ��@     ��@      �@     ��@     �@     `�@     ȅ@      �@     ��@     �@     @|@     �y@     �w@     pv@     �t@     pq@     �n@     �m@     �j@     �g@      c@      c@     �b@     ``@     �_@     �\@     �[@     @W@     �S@     @P@     �K@     �P@      L@      F@     �B@     �E@     �I@      @@     �@@      6@      4@      4@      8@      5@      1@      ,@       @      *@      .@      @      "@       @      "@      @      @      @      @      "@      @      @      �?              @      @      @       @       @      @       @      @      @      �?      �?      �?      �?      @      �?              �?              �?      �?              �?               @              �?              �?      �?              �?              �?              �?              �?      �?              �?      �?              �?              @       @      �?      @      �?      �?       @      �?       @      �?      @      @      @       @       @       @      @      @      0@      (@      &@      (@      (@      .@      *@      1@      .@      .@      (@      <@      9@      =@      =@      1@      =@      <@     �C@     �E@     �G@      I@     �F@     �L@     �P@     �W@      Q@     �V@      W@     �W@     �Z@     @_@     @c@      e@     �d@     �f@      j@     �m@     `m@     0p@      t@      w@     �v@     �y@     0|@     @~@     H�@     �@     ��@     ��@     0�@     p�@     ��@     �@     ��@     ��@     x�@     ��@     @�@     ĝ@     R�@     �@     6�@     ��@     f�@     B�@     �@     ��@     ��@     �@     Y�@     ��@     �@      �@     ��@     ,�@    �I�@     ��@     ��@    �l�@    ���@     ��@     ��@    ��@     A�@     S�@     ��@     ��@     !�@     ��@     J�@     ̰@     ȥ@     T�@      s@      ;@        
�
wo*�	   ��䣿    C�?     ��@! h�0�)�	�=5��?2�
`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'�������6�]�����[���FF�G �pz�w�7��})�l a��ߊ4F��8K�ߝ�a�Ϭ(��_�T�l׾��>M|KվK+�E���>jqs&\��>��>M|K�>�_�T�l�>�f����>��(���>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�������:�
              @      (@      $@      8@     �B@      G@     �C@     �O@     @P@      N@     �V@     �N@     @S@     �H@     �P@      L@      M@      K@      G@     �C@      P@      I@      ?@     �@@     �A@      =@     �A@      6@      4@      4@      4@      8@      5@      &@      $@      ,@      ,@      "@      @      @      @      @      @      @      @      @      @      @       @               @               @      �?       @      �?      �?      @      �?       @      @       @       @               @               @               @              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?               @              �?              �?              �?              �?              �?               @              �?      �?      �?      �?      �?       @      �?      �?       @      @              �?      @      @              �?       @      �?      @      @      @      "@      $@       @      "@      $@       @      (@      $@      "@      "@      1@      0@      ,@      6@      ;@      8@      <@      4@      A@      D@      G@      A@     �D@      F@      M@      N@     �J@     �I@      E@     �K@     @R@     �M@      J@      E@     �E@     �@@      >@      6@      &@      @      �?        
�
bc1*�	   `)�x�   ����?      @@!  �� ��?)Ӏ�V�v?2po��5sz�*QH�x�&b՞
�u�*QH�x?o��5sz?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�������:p              �?      �?              �?              �?       @      �?       @      &@      @       @        
�
bd1*�	    �<w�   `�A�?      P@!  �]i��?)�����0�?2�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp�>	� �?����=��?-Ա�L�?eiS�m�?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?�������:�              9@      @              @              �?              �?              �?      @      &@      $@       @        
�
bo*�	   ��K��   ����?     �E@!  �zƉɿ)d��� �?2��v��ab����<�A����"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m�����T}�o��5sz��N�W�m�ߤ�(g%k�a�$��{E?
����G?nK���LQ?�lDZrS?5Ucv0ed?Tw��Nof?hyO�s?&b՞
�u?����=��?���J�\�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�������:�              �?              @      @      @              �?      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              @      �?              @       @       @        ���ql%      E��	��no�A*�J

loss��_@

accuracyl	y=
�
wc1*�	   `����   ��X�?      r@!  Nw��?)��1�?2�}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A�d�\D�X=���%>��:��u�w74���82���bȬ�0�ji6�9���.��x?�x��>h�'��O�ʗ�����Zr[v��X$�z�>.��fc��>uܬ�@8?��%>��:?d�\D�X=?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?      �?              �?       @      �?      @      @      @      �?      @      @      @      @      �?      �?      @       @      �?      @      �?      �?       @      �?              @      �?      @       @      �?              @       @              �?      �?       @       @              �?               @              �?      �?              �?              �?              �?              �?              �?       @               @       @      �?      �?      �?       @              �?              �?      @              �?       @              �?      @              �?       @      @      @       @      "@      @      @      @      @      (@      &@      &@      $@      .@       @      @      *@       @      *@      *@      @      @      @        
� 
wd1*� 	   @����    ��?      A!�a �u�@)<�|��K@2�`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K��39W$:���.��fc����4[_>������m!#���
�%W����ӤP�����z!�?��K���7��[#=�؏��cR�k�e>:�AC)8g>ہkVl�p>BvŐ�r>�i����v>E'�/��x>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�������:�              @       @      0@      7@     �x@     4�@     H�@     W�@     �@     ̻@     �@    ���@     ��@     :�@    ���@    ���@    � �@     ��@     ��@     ޽@     �@     ��@     �@     l�@     Ĵ@     �@     ϱ@     ��@     2�@     N�@     x�@     l�@     ��@     ��@     ��@     X�@     d�@     ��@     ��@     �@      �@     |�@     ��@     x�@     �@     `�@     �@     �@     �@     �@     �@     �{@     �y@     �w@      w@      q@     �p@      q@     �n@     �h@     �h@      e@     �d@     �a@      ]@     �^@     �[@      ]@      ]@     �S@     @Q@     @P@      M@      I@     �K@      E@      K@      H@      B@      <@      @@      :@      6@      5@      7@      $@      4@      &@      $@      $@       @      $@       @      @       @      @      @      &@      @      @      @      �?      @      @      @       @      @      @               @      �?      �?      @      �?              �?      �?              �?               @      @              �?              �?              �?              �?              �?              �?      �?               @      �?      �?              �?      �?       @       @              �?       @      �?      @      @       @      @      @      @       @       @      @       @      *@       @      (@      ,@      $@      .@      &@      2@      *@      6@      (@      =@      <@     �A@      C@      B@      D@      H@     �P@     �P@     �I@     �O@     �T@     �L@     �T@     �U@     �[@     �]@     `a@      c@      d@     �g@     �e@     �g@     �k@     �o@     �r@     �s@      t@     �u@     P|@     �z@     �~@     �@     ��@     �@     ��@     ��@     ��@     ��@     ��@     �@     ��@     �@     ,�@     �@     ��@     ,�@     ��@     ��@     "�@     ��@     v�@     f�@     
�@     -�@     Q�@     �@     ��@     ٸ@     �@     �@     �@     O�@     ��@     ��@    �D�@    ���@     ��@    ���@     L�@    �V�@     m�@     #�@    ���@     k�@     ��@     �@     ��@     :�@     ,�@     `x@      K@      �?        
�
wo*�	   @G,��    i��?     ��@! D����)�K,��g�?2�	`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�6�]���1��a˲�O�ʗ�����Zr[v���ߊ4F��h���`�X$�z��
�}����0�6�/n�>5�"�g��>���%�>�uE����>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�������:�	              @      *@      &@      9@      D@     �E@     �G@      N@      O@      R@     @S@      P@      S@     �I@     @P@      P@      L@      G@      G@     �D@     �P@     �F@     �@@      >@     �C@      =@     �A@      4@      4@      3@      :@      1@      5@      *@      &@      "@      ,@      *@      @      @      @      @      @      @      @      @      @      �?       @      �?       @      �?      @      @      �?       @              @              @      �?      �?       @       @      �?              �?              �?              �?               @              �?              �?              �?              �?               @      �?      �?              �?      �?      �?      �?              �?      �?      �?      @      �?              �?              @      @      @       @      @      @      @      @      @      "@       @      @       @       @      "@      @      $@      &@      "@      $@      .@      2@      ,@      :@      8@      9@      :@      4@     �B@     �B@     �D@     �C@      F@     �C@      M@     �O@      J@      I@     �E@     �L@     �Q@      K@     �J@      C@     �G@     �B@      ;@      7@      *@      @      �?        
�
bc1*�	    wy�   ��:�?      @@!  �z���?)U����x?2po��5sz�*QH�x�&b՞
�u�*QH�x?o��5sz?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?�������:p              �?      �?              �?               @       @      �?      @      @      "@      �?        
�
bd1*�	    \w�   @��?      P@!  �����?)��A� T�?2�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp�>	� �?����=��?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?�������:�              9@      @              @              �?              �?              �?      @      @      *@      @        
�
bo*�	   @ ^��   ���?     �E@!  ��`gʿ).̎_<��?2��v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m�����T}�o��5sz�uWy��r�;8�clp�1��a˲?6�]��?d�\D�X=?���#@?5Ucv0ed?Tw��Nof?hyO�s?&b՞
�u?����=��?���J�\�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?              �?      "@      @      @      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?               @      �?      �?              @      @      �?        2�d2�%       �	 &�oo�A*�K

loss��_@

accuracy�7x=
�
wc1*�	   �X���   `�}�?      r@!  O���?)�%'�\�?2�}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E����#@�d�\D�X=���%>��:��u�w74���82���VlQ.��7Kaa+�I�I�)�(�豪}0ڰ>��n����>x?�x�?��d�r?�vV�R9?��ڋ?�S�F !?�[^:��"?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?      �?              �?       @       @      @      @      @       @      @      @      @      @      �?      �?      @       @       @      @      �?               @      �?      �?      @      �?       @      @      �?      �?      �?      �?       @      �?      �?      @      �?              �?      �?              �?              �?      �?              �?              �?              �?              �?              �?               @      �?      @              �?      �?      @               @              �?       @      �?       @              �?      @              �?       @       @      @       @      @      @      @      @      @      $@      *@      &@      $@      (@      "@      "@      *@       @      (@      ,@      @      @      @        
�!
wd1*�!	   �a@��   @�?      A!i����@)U�
�� L@2��/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�������������?�ګ�;9��R���5�L�����]������|�~���u��gr��R%������39W$:���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��E'�/��x��i����v�w&���qa�d�V�_�w&���qa>�����0c>f^��`{>�����~>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�������:�              �?       @      @      1@      6@     �y@     0�@     v�@     f�@     ߷@     ѻ@     �@    ���@     �@    ��@    ���@    ���@     �@     b�@     a�@     Խ@     ۻ@     G�@     ظ@     y�@     ��@     ߲@     ޱ@     K�@     ��@     B�@     >�@     $�@     ��@     z�@     r�@     П@     ��@     (�@     ��@     $�@     x�@     ��@     �@     h�@     ��@     (�@     ��@     ��@     h�@     0�@      �@     �|@     �x@     �x@     pu@     0r@     �p@     �m@     �l@     �l@      f@     �c@     `f@      ]@      a@      `@      [@      X@      [@     �X@     �Q@     �P@     �S@      H@     �D@     �J@      B@      B@      @@      ;@      <@      5@      ;@      ,@      =@      2@      &@      (@      (@      .@      (@      @      &@       @      (@      @       @      @      @      @       @      @      @      @       @       @      @       @      �?       @              @      �?               @      �?               @      �?              �?              �?      �?      �?       @              �?              �?              �?              �?              �?              �?       @              �?               @              �?      �?       @      �?              �?               @      @      @      @      @      @       @       @       @      @      @      @       @      @      @       @      "@      (@      "@      &@      *@      4@      .@      6@      7@      2@      7@      7@      =@     �E@     �@@      H@      =@     �O@     �L@      N@      O@     @S@     @S@     �X@     �V@     �Z@     @\@     @a@      `@     �c@     �c@     @h@      h@     �o@      o@     �o@     Ps@      u@     �w@     �{@     `{@      @     ��@     ��@     �@     �@     ��@     ��@     X�@     ��@     �@     0�@     X�@     �@     ��@     ��@     ؟@     �@     
�@     p�@     ��@     �@     Z�@     2�@     ,�@     ��@     J�@     ��@     ��@     ��@      �@      �@     D�@     ��@     ��@     Z�@     }�@     ��@     ��@     P�@     m�@     ��@    �8�@     >�@     ��@     �@     �@     ��@     ��@     ��@     �~@     �T@      @        
�
wo*�	    k��    ��?     ��@! ��7�;�)��{����?2�	�/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲�>�?�s���O�ʗ���pz�w�7��})�l a�8K�ߝ�a�Ϭ(�G&�$��5�"�g���>�?�s��>�FF�G ?��[�?1��a˲?��d�r?�5�i}1?�T7��?�vV�R9?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�������:�	              �?      @      *@      (@      ;@     �C@     �F@      F@      O@     �O@     �R@     �R@     @Q@     @Q@      L@      P@      P@     �L@     �G@      F@     �C@     @Q@     �F@      A@      >@     �B@      ?@      <@      8@      6@      3@      8@      .@      6@      ,@      $@      ,@       @      *@      @      &@      @      @      @      @      @      @      @      @       @       @      �?      �?       @              @              �?       @      @              @      �?      �?      �?      �?               @              �?              �?      �?      �?              �?              �?              @              �?               @              �?               @              �?              �?      �?      �?      �?               @      �?       @      �?      �?       @      @      @      @       @      �?      @      @      @       @       @      @      "@      &@       @      $@      @      $@      &@      ,@      2@      0@      9@      =@      6@      5@      ;@      >@     �A@     �E@      D@     �F@      D@     �L@     �P@     �F@      M@     �C@      M@     �P@     �L@      J@      C@     �H@      B@      :@      :@      *@      @      �?        
�
bc1*�	   ��7y�    �v�?      @@!  �����?)#�}���z?2xo��5sz�*QH�x�&b՞
�u�o��5sz?���T}?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�������:x              �?      �?              �?               @      �?      �?      @      "@      @      @      �?        
�
bd1*�	   `�xw�    ���?      P@!   �a��?)*0�G�m�?2x*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp�����=��?���J�\�?eiS�m�?#�+(�ŉ?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?�������:x              9@      @              @              �?              �?              @       @      &@      @        
�
bo*�	   �e��   �Yќ?     �E@!   )�<˿)����?2��/����v��ab��}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ����T}�o��5sz�&b՞
�u�hyO�s��T���C��!�A�I�I�)�(�+A�F�&�5Ucv0ed?Tw��Nof?hyO�s?&b՞
�u?����=��?���J�\�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?              �?      $@      @      @      �?               @              �?              �?              �?              �?              �?              �?              �?              �?              �?       @      �?              @      @      �?        }���%      [0$�	�y~po�A *�K

loss��_@

accuracy�ew=
�
wc1*�	    ����   ����?      r@! �R����?).1��q��?2�}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G����#@�d�\D�X=���%>��:�uܬ�@8���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$�I��P=��pz�w�7��>h�'�?x?�x�?�S�F !?�[^:��"?U�4@@�$?�7Kaa+?��VlQ.?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?      �?      �?               @       @      @      @      @       @      @      @      @      @              �?      @      �?       @       @       @               @      �?       @              @       @      @              �?      �?      �?      @              �?      @              �?              �?              �?      �?              �?              �?              �?              �?      �?              �?              �?               @      �?      �?       @              �?       @      �?               @              �?       @               @      �?      �?      @              �?       @       @      @       @      @       @      @      @      @      $@      &@      *@       @      *@      $@      "@      (@      "@      (@      ,@      @      @      @        
� 
wd1*� 	   @���    |x�?      A!v�@);��9�~L@2��/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]�����MZ��K���u��gr��R%������39W$:���X$�z��
�}�����4[_>������m!#���
�%W��u��6
��K���7��f^��`{>�����~>[#=�؏�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�u��gr�>�MZ��K�>���]���>�5�L�>;9��R�>���?�ګ>����>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�������:�              �?      @      @      3@      8@      {@     0�@     ��@     w�@     �@     �@     ��@    ���@     d�@     �@     ��@    �o�@     %�@     4�@     M�@     ��@     Ż@      �@     ʸ@     q�@     ��@     �@     ˱@      �@     ڬ@     N�@     ܨ@     >�@     b�@     ��@     ��@     �@     D�@     \�@     ��@     ��@     ,�@     ؑ@     Џ@     @�@     H�@     ؉@     x�@     ��@     ��@     ��@      @     �{@     �z@     `v@     @u@     r@     �q@     @n@      n@     �i@      f@     @e@     �d@     �a@     @[@      `@     �Z@      V@      [@     �U@      K@     �P@     �I@     �J@     �J@     �F@      F@     �E@     �C@      :@      :@      9@      6@      3@      0@      5@      4@      ,@      *@      &@      ,@      *@      (@      @      @      "@      @       @      @      �?      @      @      @      @       @      @      @      �?       @      �?       @      @      �?      �?      �?      �?               @               @              �?       @               @              �?              �?      �?              �?      �?      �?              �?              @       @       @       @              �?       @       @      �?      �?      @       @       @       @      @      @      @      @      @      @       @      @      @      &@      $@      2@      ,@      (@      1@      (@      1@      9@      <@      =@      <@      6@      A@     �C@      E@      K@     �I@     �I@     @R@     �N@     @W@      Q@     @X@     �T@     �\@     @_@      ]@     @^@     �b@      c@     @f@     @j@      k@     @q@     q@     `t@     v@     w@     �z@     �|@     �}@     @     ��@     ��@     0�@     8�@     Њ@     ��@     ܐ@     <�@     |�@     ��@     ė@     l�@      �@     �@     L�@     ��@     ��@     ��@     ��@     �@     ֯@     ߰@     ��@     0�@     ��@     ��@     ��@     Ǽ@     ��@     '�@     ��@    ���@    �r�@     N�@     ��@     ��@     J�@     r�@    ���@    ��@     ��@     ��@     M�@     z�@     s�@     \�@     T�@     ��@     @^@      @        
�
wo*�	   `����   @pc�?     ��@! X��;��)�KF�$�?2�	�/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[��O�ʗ�����Zr[v���f�����uE�����uE����>�f����>�ߊ4F��>})�l a�>O�ʗ��>>�?�s��>�FF�G ?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�������:�	              �?       @      (@      1@      7@     �C@      F@     �G@     �O@      O@     �R@     �R@     @P@     @Q@     �L@      R@     �N@     �L@     �G@     �F@     �E@     �L@     �G@     �@@      A@     �D@      >@      3@      =@      8@      1@      4@      3@      4@      &@      *@      *@      &@      0@      @      @      @       @      @      @      @       @      @      @       @      @               @       @      �?              �?              @      �?              �?              �?              @      �?       @              �?      �?      �?              �?              �?              �?              �?              �?              �?      �?              �?      �?      �?      @      �?      �?      �?              �?      �?      @      @      �?      @       @      �?      @      @      @      @      @      @      @      @      @      "@      @      (@      "@      "@      @      *@      ,@      *@      ,@      2@      ;@      7@      <@      7@      4@      <@     �E@      B@      E@     �F@      D@      I@     @R@     �H@      J@     �E@     �J@     �P@     �N@      J@     �A@      G@     �C@      :@      =@      *@      @       @        
�
bc1*�	   �0Ty�    %��?      @@!   /���?)3���M|?2xo��5sz�*QH�x�&b՞
�u�o��5sz?���T}?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�������:x              �?      �?              �?              �?      �?       @       @      $@      @      @      �?        
�
bd1*�	    y�w�    �%�?      P@!  ��^�?)�ߝcq�?2�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp�����=��?���J�\�?eiS�m�?#�+(�ŉ?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�������:�              :@      @              @              �?              �?              @      @       @      $@       @        
�
bo*�	   @r`��   �ȝ?     �E@!  ���̿)g���ܐ?2��/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���7c_XY��#�+(�ŉ�eiS�m��>	� �����T}�*QH�x�&b՞
�u��lDZrS�nK���LQ�k�1^�sO�IcD���L�5Ucv0ed?Tw��Nof?hyO�s?&b՞
�u?���J�\�?-Ա�L�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?              �?       @      $@      @      @              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?       @      �?              @       @       @        �x��&      SN�R	<�2qo�A!*�L

loss�`@

accuracyڬz=
�
wc1*�	   @0���   ���?      r@! �����?)���~�?2�}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I����#@�d�\D�X=���%>��:�uܬ�@8���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&��FF�G ?��[�?�T7��?�vV�R9?�S�F !?�[^:��"?�7Kaa+?��VlQ.?��%�V6?uܬ�@8?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?      �?      �?      �?      �?      @      @      @      @       @      @      @      @      @               @      @      �?       @       @       @               @      �?       @              @      @      @      �?              �?      @      �?      �?      @              �?              �?              �?              �?              �?              �?              �?               @              �?              �?               @      �?      �?              �?              @              �?       @               @      �?               @      �?      �?      @               @       @       @      @       @      @      "@      @      @      @      $@      (@      (@       @      *@      $@      "@      (@      "@      (@      ,@      @      @      @        
� 
wd1*� 	   @v���   @W�?      A! }46�!�@)$K#xn�L@2��/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~��.��fc���X$�z��
�}����[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t���8"uH���Ő�;F���Ő�;F>��8"uH>d�V�_>w&���qa>w`f���n>ہkVl�p>���m!#�>�4[_>��>R%�����>�u��gr�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�������:�              �?      @      @      3@      8@     `|@     D�@     Ԧ@     w�@     ��@     ��@     �@     ��@     f�@    ��@    ���@    �p�@     1�@     .�@     N�@     ~�@     ʻ@     Ϲ@     ��@     ��@     K�@     Ѳ@     �@     ��@     (�@     �@     ��@     ��@     H�@     ��@     ��@     ��@     t�@     ș@     ��@     ؔ@     ��@     |�@     l�@     �@     H�@     8�@     ��@     ��@     ��@     �@     p|@     �|@     �z@     �v@     u@     Pr@     0q@     �n@     @k@     �i@     �g@      d@     �e@     �_@     @`@     �^@     �X@     @\@     �W@     �X@     @R@      O@     �M@     �L@     �C@     �C@      C@      B@      A@      A@      6@      5@      &@      9@      7@      4@      &@      .@      ,@      *@      @      *@       @      "@      @      &@       @      "@      @      @      @       @       @      @      @      @      @               @              �?       @      �?               @      �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              @              �?               @              @               @      �?      @       @      �?      @      �?              @      @      @      @      @      @       @      @      "@      "@      &@      @      &@      .@      &@      @      2@      (@      5@      *@      4@      4@      0@      8@      <@      ?@      C@      A@      I@     �J@     �I@     �I@     �L@     �N@     @Q@     �S@      W@     �V@      Y@     @]@     �\@     �b@     �b@     `g@      f@     �f@     @j@     �l@     �r@     0t@      s@      w@     �z@     @{@     @@     ��@     ��@     ��@     ��@     ��@     ��@     ��@     ��@     ؑ@     x�@     t�@     �@     ̚@     Ԝ@     �@     N�@     Ƣ@     ĥ@     ��@     ��@     J�@     �@     	�@     ��@     ;�@     ��@     J�@     պ@     ��@     @�@     -�@     o�@    ���@     x�@    �1�@     ��@     \�@    �<�@     ��@     ��@     �@     ��@     ��@     ��@     ��@     ۳@     �@     ��@     ��@      c@      &@        
�
wo*�	   �zǤ�    ���?     ��@! ��!��)pF��9a�?2�	�/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"�ji6�9���.����ڋ��vV�R9��5�i}1���d�r�x?�x��>h�'�������6�]���1��a˲���[��>�?�s���O�ʗ����ߊ4F��h���`�8K�ߝ�a�Ϭ(龢f�����uE���⾙ѩ�-߾E��a�Wܾjqs&\��>��~]�[�>��>M|K�>O�ʗ��>>�?�s��>6�]��?����?f�ʜ�7
?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�������:�	               @       @      &@      2@      8@     �C@     �F@      G@     �O@     @P@     @R@     @S@     �O@     �P@      M@     @R@     �L@      O@      H@      E@     �F@      L@     �G@     �@@      A@     �D@      ;@      ;@      9@      9@      .@      .@      7@      2@      "@      .@       @      (@      0@      &@      @      @      "@      @      �?      @       @      @      �?      �?      @      @       @      �?              �?      �?      �?      �?      �?               @      �?       @              �?      �?       @               @              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?      �?               @      @      �?       @      �?               @       @      �?              @      @      @      @      �?      @      @              @      @      @      @      @      "@      @      $@       @       @      &@       @      .@      0@      ,@      .@      ;@      <@      7@      9@      1@      ?@      D@     �C@      D@     �F@     �D@     �H@     @R@     �I@     �J@     �D@      J@     �O@     �P@      J@     �A@      F@     �C@      :@      =@      ,@      @      @        
�
bc1*�	   � ny�   �̈́�?      @@!  �Ba��?)��?@�}?2�o��5sz�*QH�x�&b՞
�u�o��5sz?���T}?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?�������:�              �?      �?              �?              �?      �?       @      �?       @      @      "@              �?        
�
bd1*�	   `�w�   �ꯗ?      P@!   P�?)�-7ZV�?2�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp�����=��?���J�\�?eiS�m�?#�+(�ŉ?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�������:�              ;@       @              @              �?              �?              @      @      "@       @      @        
�
bo*�	   ��&��   @Þ?     �E@!  ����̿)#�X�\��?2��/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���7c_XY��#�+(�ŉ�eiS�m��>	� �����T}�o��5sz�*QH�x��m9�H�[���bB�SY�5Ucv0ed?Tw��Nof?hyO�s?&b՞
�u?���J�\�?-Ա�L�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�������:�              �?              �?      @      $@      @      @              �?      �?              �?              �?               @              �?              �?              �?              �?              �?       @      �?              �?       @      @      �?        ���|&      cT?�	Wy�qo�A"*�L

loss��_@

accuracy�7x=
�
wc1*�	   �T���   � }�?      r@!  s�#�?)9hT�?2�}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��!�A����#@���%>��:�uܬ�@8��u�w74���82��T7����5�i}1�O�ʗ��>>�?�s��>f�ʜ�7
?>h�'�?ji6�9�?�S�F !?�7Kaa+?��VlQ.?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?      �?      �?      �?      �?      @      @      @      @      @      @      @      @      @               @      @      �?       @      @      �?      �?      �?      @              @       @      @      �?      �?       @      �?       @       @       @      �?              �?               @              �?               @              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?               @      �?              �?       @              �?      �?      �?               @      �?      �?       @       @               @      @       @      @       @       @      @      @      @      @      &@      (@      $@      $@      *@       @      (@      &@       @      (@      *@      @      @      @        
�!
wd1*�!	   �����   ��Q�?      A!L��c�@)U���L@2��/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc����4[_>������m!#����ӤP�����z!�?��u��6
��K���7��[#=�؏��/�p`B�p��Dp�@�w&���qa>�����0c>T�L<�>��z!�?�>��ӤP��>�
�%W�>�4[_>��>
�}���>.��fc��>39W$:��>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?�������:�              �?      @       @      3@      9@     �}@     l�@     ��@     ��@     �@     �@     �@     ��@     ~�@      �@     ��@     ��@     #�@     M�@     A�@     ��@     ��@     
�@     ��@     ��@     W�@     �@     ��@     2�@     ��@     P�@     ��@     4�@     �@     T�@     0�@     ��@     ě@     ��@     �@     $�@     �@     h�@     �@     �@     @�@     ��@      �@     ��@     �@      @     `@     �}@     �x@     �u@     �t@     0u@     pp@     �n@     �j@     �g@      i@     �d@     @d@     �c@      ^@     �\@     �_@      [@     @Y@      X@      U@      Q@      H@     �F@      G@     �C@     �G@     �H@      A@      ;@      4@      4@      7@      <@      5@      1@      ,@      .@      2@      "@      $@      @      $@      @       @       @      @       @      @      @              @      �?      @      �?      @      @       @      �?              �?      @       @      �?      �?      �?               @      @              �?      �?              �?               @              �?      �?              �?              �?               @              �?              �?              �?              �?              @       @       @      �?      @       @      �?      @               @      �?      �?       @      @      @      @       @      @      $@       @      @      @      "@      @      "@      ,@      .@      $@      *@      *@      .@      4@      0@     �@@      ;@      0@      6@      7@      A@     �B@     �C@     �B@      K@     �I@     �N@      N@     @R@     �Q@     @W@     �W@      Y@      X@     �_@     �c@     �e@     �b@     �f@     @k@      h@     Pp@     Pr@     �s@     pt@     �x@     �x@      {@     �}@     @�@     ��@     p�@     ��@     ��@     ��@     Ȏ@     ��@     ��@     t�@     ��@     ��@     �@     ��@     ��@     J�@     �@     �@     ��@     ��@     Ԭ@     �@     �@     ��@     ��@     ��@     C�@     ��@     ��@     l�@     �@    �Q�@    ���@    �]�@     .�@     ��@     '�@    �I�@    �[�@    ���@    ��@    ���@     ��@     Ѿ@     ��@     ��@     N�@     ��@      �@     `e@      2@      �?        
�
wo*�	   �Mޤ�   ��ޠ?     ��@! ഄ���)5Y����?2�	�/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7���f�ʜ�7
������6�]���1��a˲���[���FF�G ���Zr[v��I��P=���iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿ
�/eq
Ⱦ����ž
�/eq
�>;�"�q�>��(���>a�Ϭ(�>��[�?1��a˲?��d�r?�5�i}1?�T7��?�vV�R9?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?�������:�	               @       @      &@      2@      :@      B@     �F@     �G@     �O@     @Q@     �Q@      T@      M@      Q@     @P@     @P@     �N@      O@      H@     �D@      F@      K@      J@      =@      B@      B@     �@@      6@      ?@      7@      .@      1@      1@      2@      *@      *@      &@      "@      &@      (@       @      @      @      @      @      "@       @              �?       @      @      @      @      �?              �?      @              �?      �?      @               @              �?       @              �?      �?              �?       @              �?               @      �?      �?              �?              �?              �?              �?              �?       @      @               @      �?      �?              �?      �?       @      @      @       @              @       @       @      @      @       @      @      @      &@      @      "@      $@      "@       @      "@      (@      (@      3@      *@      ;@      ;@      <@      6@      8@      9@      C@     �D@     �C@     �H@     �B@      I@     �Q@      M@      I@      D@      J@      P@     �P@      I@      B@      E@      E@      8@      @@      *@      @       @      �?        
�
bc1*�	   `��y�    �/�?      @@!  ���,�?)cL��?2xo��5sz�*QH�x�&b՞
�u����T}?>	� �?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?�������:x              �?      �?              �?               @      �?      �?      @      &@      "@              �?        
�
bd1*�	   @��w�   �0+�?      P@!   2Z��?))���?2�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp�����=��?���J�\�?eiS�m�?#�+(�ŉ?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�������:�              ;@       @              @              �?              �?               @      @      $@      "@      @        
�
bo*�	   `����   �L��?     �E@!  �¤oͿ)�d��T�?2��/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���7c_XY��#�+(�ŉ�eiS�m��>	� �����T}�o��5sz�*QH�x�5Ucv0ed����%��b��l�P�`�5Ucv0ed?Tw��Nof?uWy��r?hyO�s?���J�\�?-Ա�L�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?}Y�4j�?��<�A��?�v��ab�?�/��?�������:�              �?              �?      @       @      @      @              �?      �?              �?              �?              �?      �?              �?              �?              �?              �?              �?       @      �?              @      @      �?        �L�j|&      cT?�	���ro�A#*�L

loss�_@

accuracy��t=
�
wc1*�	   ����    BS�?      r@!  f��K�?)Ao�F�
�?2�}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:���82���bȬ�0�U�4@@�$��[^:��"���ڋ��vV�R9�>�?�s���O�ʗ���O�ʗ��>>�?�s��>�vV�R9?��ڋ?ji6�9�?�S�F !?�7Kaa+?��VlQ.?��82?�u�w74?��%>��:?d�\D�X=?���#@?�!�A?�T���C?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?      �?       @              �?      @      @      @      @      @      @      @      @       @              @       @       @      @       @              �?      @       @       @      @       @       @       @       @      �?      @       @              �?              �?      �?              �?               @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?               @              �?      �?      �?      �?              �?       @               @      �?      �?      �?      @              �?       @      �?      @      @      �?      "@      @      @      @      @      &@      &@      "@      $@      .@       @      $@      &@      "@      (@      (@      @      @      @        
�!
wd1*�!	   �z5��   ����?      A!z����ΐ@)�{�E�L@2��/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ��5�L�����]������|�~���MZ��K��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���i����v��H5�8�t�cR�k�e������0c���8"uH>6��>?�J>�����0c>cR�k�e>�4[_>��>
�}���>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?�������:�              �?      @       @      5@      =@     �~@     ��@     �@     ̳@     �@     G�@     5�@    ���@    ���@     $�@     ��@     ��@     4�@     X�@     ��@     ��@     Ļ@     
�@     ��@     ��@     ��@     β@     ��@     I�@     Ԭ@     T�@     ��@     @�@     �@     ��@     �@     ��@     ؜@     ��@     d�@     �@     p�@     <�@     Ȑ@     ��@     ��@     0�@     (�@     Ȅ@      �@     ��@     �@     �{@      x@      v@     0w@     �r@      q@      p@     �j@     `h@     @g@     �d@      e@     �a@     �^@      ^@     @]@     @[@      Z@     @S@     �S@      L@      L@     �L@      G@     �D@      F@      D@      A@      A@      ;@     �@@      7@      3@      <@      :@      2@      1@      2@      ,@      @      *@       @       @      @       @      $@      @      @      @       @      @       @      @      @      @      @      �?              �?               @      @               @       @      �?              �?      �?      �?              �?      �?              �?              �?              �?              �?              �?              �?      �?       @               @      �?              �?              �?      @       @              �?      @      �?      �?      @      @       @      �?      @      @      @      @      @      *@      $@      $@      *@       @      $@      *@      *@      1@      (@      3@      *@      3@      5@      >@      :@      7@      A@      B@      B@     �C@     �I@     �D@      P@     �N@      T@     �S@     �S@      X@     �Y@     �[@      a@      `@     �a@     @e@      e@      i@      i@      q@      r@     s@     �t@     0w@      z@     Pz@     �@     ��@     ��@     �@     @�@     ȉ@     ؋@     ��@     ��@     ��@     �@     �@     ��@      �@     ��@     :�@     H�@     �@     ��@     ��@     `�@     ��@     ޮ@     �@     �@     ��@     {�@     ��@     f�@     ��@     ˿@    ���@    �A�@     ��@     +�@     8�@    ���@    ��@    �5�@     E�@    �v�@     ��@    ���@    ���@     ��@     4�@     �@     �@     ��@     ��@     �g@      :@       @        
�
wo*�	   @v餿    .�?     ��@! g�!��)�pU��?2�	�/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲�pz�w�7��})�l a��ߊ4F��h���`�a�Ϭ(���(�����~]�[Ӿjqs&\�Ѿ
�/eq
Ⱦ����ža�Ϭ(�>8K�ߝ�>pz�w�7�>I��P=�>>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?�������:�	               @       @      &@      2@      8@     �D@     �D@      I@      N@     �R@     @P@     @S@     @P@      Q@     �N@     @P@     @P@      O@     �F@     �D@      G@     �M@      H@      9@     �A@      C@     �A@      5@      @@      9@      0@      1@      0@      0@      (@      0@      @       @      $@      0@      @      @      @      @      �?      @      @       @       @      �?      @      @       @               @       @      �?      �?      �?      �?       @      �?      �?      @      �?               @       @      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?      �?      �?              @      �?              �?      @      �?      @      @              @      @      @       @      @      @       @      @      @       @      (@      "@      $@      @      @      @      *@      0@      0@      (@      >@      8@      <@      7@      8@      :@      E@     �C@     �C@     �F@     �E@      H@     �Q@     �M@      I@     �B@     �L@      N@     �O@      J@      C@     �D@     �D@      :@      ?@      ,@      @       @      �?        
�
bc1*�	   �F�y�    {��?      @@!   7���?)=XF}��?2xo��5sz�*QH�x�&b՞
�u����T}?>	� �?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?�������:x              �?      �?              �?               @      �?      �?      @      $@      $@              �?        
�
bd1*�	   @1�w�   �d��?      P@!   ����?)`|G6F��?2�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp�����=��?���J�\�?eiS�m�?#�+(�ŉ?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�������:�              ;@       @              @              �?              �?               @       @      &@      "@      @        
�
bo*�	   @&���   �E`�?     �E@!  �m�ο)g>���?2��uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��eiS�m��-Ա�L������=���>	� �����T}�o��5sz�ߤ�(g%k�P}���h�5Ucv0ed����%��b�5Ucv0ed?Tw��Nof?uWy��r?hyO�s?���J�\�?-Ա�L�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?}Y�4j�?��<�A��?�v��ab�?�/��?�������:�              �?              �?              @      @      @      @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?       @      �?              @       @       @        ���O.'      i�?�	�Rso�A$*�N

loss�}_@

accuracyڬz=
�
wc1*�	    H���   ��?      r@!  ��t;�?)NWN���?2�}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@���%>��:�uܬ�@8��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�U�4@@�$��[^:��"��S�F !�ji6�9��5�"�g���0�6�/n��>h�'�?x?�x�?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?�u�w74?��%�V6?uܬ�@8?��%>��:?�!�A?�T���C?a�$��{E?
����G?�qU���I?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?       @      �?              �?      @      @      @      @       @      @      @      @      �?              @       @       @      @       @      �?      �?      @      @      @      �?      �?      @       @       @      @              �?       @              �?      �?      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?              �?              �?               @              �?      �?              �?      �?              �?       @       @              �?      �?      �?              �?      @       @      @      �?      @      "@      @      @      @      @      $@      ,@       @      "@      &@      $@      $@      "@      &@      $@      (@      @      @      @        
�!
wd1*�!	   ��p��   `��?      A!l��c�`�@)Qv���L@2��/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]�����MZ��K���u��gr��R%������.��fc���X$�z��
�}�����4[_>������m!#���
�%W����z!�?��T�L<��u��6
��K���7���H5�8�t�BvŐ�r�w`f���n�=�.^ol�ڿ�ɓ�i�d�V�_>w&���qa>�4[_>��>
�}���>X$�z�>.��fc��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?�������:�               @      @      (@      2@      D@     �@     �@     6�@     ѳ@     H�@     ��@     ^�@     �@    ���@     2�@    ��@     ��@    �e�@     X�@     ��@     ��@     �@     `�@     ��@     ��@     z�@     �@     ̱@     i�@     ��@     D�@     ��@     $�@     8�@      �@     r�@     �@     ؛@     ̙@     Ԗ@     ��@     ē@     4�@     \�@     ��@      �@     `�@     ؅@     �@     ��@      �@     �}@     |@     �x@     �w@     �t@     �q@     �p@     p@      l@      i@     �f@     �b@     @d@      a@      [@      [@     @Z@     �X@     �[@     @U@     �T@     �Q@      M@      K@      K@      F@      J@      C@      D@     �@@      6@      9@      ;@      0@      4@      8@      .@      .@      .@      *@      &@      "@      @      @      @      &@      @      "@      @      @      @      @      @      @      �?      @      �?      @              @      �?      �?      @       @      �?      �?               @      �?               @      �?      �?              �?              �?              �?              �?              �?      �?              �?               @              �?              �?      �?      �?      @              �?              �?              �?       @      �?       @      �?      @      @      �?      @      �?      @      @      @      &@       @       @      *@       @      *@      $@      .@      *@      ,@      3@      .@      $@      *@      7@      9@      7@     �A@     �C@      E@     �I@     �J@      E@      Q@      P@      P@     �T@     �V@     �Z@     �X@     @Z@     �`@     �a@     �c@     �d@     @j@      g@     �m@     �m@     0r@     �s@     �u@     �x@     �y@     pz@     �~@     `�@     ؀@     ��@     Ѕ@     �@     �@     ��@     @�@     �@     4�@     �@      �@     0�@     ��@     8�@     L�@     ��@     X�@     �@     p�@     ҭ@     �@      �@     ǲ@     ��@     z�@     ��@     ��@     ��@     ��@    �
�@     *�@    ���@     �@    �Q�@    ���@    ���@    �2�@    ��@    �J�@     ��@    �3�@    ���@     $�@     �@     ��@     ��@     `�@     X�@     `j@     �A@       @        
�
wo*�	   `�뤿   ��|�?     ��@! ��G���)�n�{���?2�	�/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.���vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
�O�ʗ�����Zr[v��8K�ߝ�a�Ϭ(���~��¾�[�=�k��豪}0ڰ�������ѩ�-�>���%�>a�Ϭ(�>8K�ߝ�>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?�������:�	               @       @      &@      2@      6@      E@      D@     �I@      N@      R@     �P@     �S@     @P@      Q@     �O@     �N@     �O@      Q@     �D@     �E@      F@     �N@      F@      =@      C@      B@      ?@      :@      ?@      8@      0@      2@      3@      ,@      "@      .@      "@      (@      "@      (@       @      @      $@      @      �?       @      @       @              @      @      @              @       @      �?              �?       @       @      �?       @      �?              �?              �?      �?              �?              �?              �?              �?              �?              �?               @              �?              �?      �?              �?              �?      �?              �?      �?              �?               @               @      �?              @      �?      @      @      �?       @               @      @      @      @      �?      @      $@      @      *@      $@      @       @      @      @      ,@      &@      0@      2@      6@      ;@      <@      4@      <@      ?@     �D@     �A@      D@      D@     �E@     �J@     �Q@      N@      J@      A@     �M@      M@      P@     �J@      A@      F@      B@     �@@      =@      ,@      @       @      �?        
�
bc1*�	   �߮y�   �d�?      @@!  ����?)l��A�W�?2�o��5sz�*QH�x�&b՞
�u�>	� �?����=��?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�������:�              �?      �?              �?               @      �?      �?      @      "@      @      @              �?        
�
bd1*�	   �	�w�   �� �?      P@!   �>�?) ܷ��3�?2�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp�����=��?���J�\�?eiS�m�?#�+(�ŉ?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�������:�              ;@       @              @              �?              �?               @       @       @      $@      @        
�
bo*�	   �	a��    eߠ?     �E@!   jz�ο)�}IŨ��?2��uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��eiS�m��-Ա�L������=���>	� �����T}�o��5sz�;8�clp��N�W�m�P}���h�Tw��Nof�5Ucv0ed?Tw��Nof?;8�clp?uWy��r?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�#�h/�?���&�?�Rc�ݒ?^�S���?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?�������:�              �?              �?              &@      @       @      @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?       @              @      �?       @      �?        m�N&      nJ��	D�to�A%*�L

loss@X_@

accuracy�7x=
�
wc1*�	   �*���   �ל?      r@!  ��?)b���C�?2�}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:���bȬ�0���VlQ.��7Kaa+�x?�x��>h�'��f�ʜ�7
������>h�'�?x?�x�?ji6�9�?�S�F !?+A�F�&?I�I�)�(?��bȬ�0?��82?�u�w74?d�\D�X=?���#@?�!�A?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?       @      �?               @       @      @      @       @      @      @       @      @              �?      @      @       @      @       @      �?       @      @      @       @              @       @      @              �?      �?      �?              @      �?      �?      �?              �?      �?       @       @              �?      �?              �?              �?              �?              �?              �?              �?      �?              �?      �?              �?      �?              �?      �?       @               @              �?      �?      �?       @      @       @       @      @      @      @      @       @      @      @      (@      *@       @      $@      $@      $@      $@      "@      $@      $@      &@      @      @       @        
�!
wd1*�!	   �����   ��}�?      A!��Bx9��@)���ˠL@2��/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����;9��R���5�L����|�~���MZ��K���u��gr��R%������
�}�����4[_>���K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�w`f���n�=�.^ol�4�j�6Z�Fixі�W�=�.^ol>w`f���n>�i����v>E'�/��x>[#=�؏�>K���7�>u��6
�>T�L<�>��ӤP��>�
�%W�>
�}���>X$�z�>.��fc��>39W$:��>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?�������:�               @      @      (@      4@     �H@     ��@     X�@     V�@     ��@     u�@     Ǽ@     ��@    �,�@    ���@     \�@     ;�@    ���@    ���@    �_�@     ��@     �@     (�@     V�@     ׸@     ��@     ��@     ��@     ױ@     k�@     ��@     ��@     �@     6�@     �@     ��@     *�@     Z�@     ��@     ,�@     ��@     (�@     h�@     ��@     t�@     P�@     ��@     �@     ��@     P�@     X�@     ؀@     �~@     0z@     �u@     Px@     �s@      s@     �p@     `p@      k@      k@     �f@     �c@      e@     �`@     @]@     �\@     @^@     �Z@     �V@     �S@      R@     �Q@      M@     �M@      I@     �D@      E@      E@      >@      :@      :@      @@      1@      :@      .@      7@      (@      *@      &@      @      $@      &@       @      &@      $@      @      @      @      @      @      @      @       @      @      @      @       @               @       @               @              �?      �?      �?              �?              �?              �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              @      �?      �?              �?              �?      �?      @       @      @      @      @       @      @      @      @      &@      @      @      @      @      ,@      3@      *@      ,@      2@      ,@      ,@      5@      ;@      2@      9@      3@      9@     �B@      C@      C@     �J@      F@     �H@     �Q@     �R@     @S@     �S@     �U@     @V@     �Y@      [@     �a@      b@     �a@     �d@     �f@      l@     �k@     0q@     �q@     �r@     �t@     �w@     @x@      |@     p�@     �@     �@     �@     ��@     ��@     @�@     ��@     ��@     x�@     ��@     �@     З@     ��@     ��@     p�@     4�@     (�@     h�@     &�@     ��@     h�@     n�@     �@     ��@     ٴ@     0�@     ��@     �@     J�@     ߿@     ��@     2�@     ��@    ���@    �%�@    ���@     ��@    ���@    ���@    �D�@    �h�@     ��@    �z�@     ƽ@     A�@     U�@     �@     ̛@     �@     �j@     �F@       @        
�
wo*�	   `�礿    �ۡ?     ��@! ���Jp�){�TI=t�?2�	�/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !��vV�R9��T7����5�i}1���d�r�x?�x����[���FF�G ��5�L�>;9��R�>a�Ϭ(�>8K�ߝ�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?�.�?ji6�9�?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?�������:�	               @      @      (@      2@      5@      E@     �D@     �F@      P@     �R@     �P@      S@      Q@      Q@     �N@     �N@     �M@     @Q@      E@      G@     �G@     �M@      G@      :@     �C@     �A@     �@@      8@      >@      5@      2@      3@      5@      ,@      "@      .@      "@      "@      "@      ,@      @      @      @      @      @      @      @       @      �?      @      @      @      �?       @       @              �?       @      @      �?      �?               @       @       @      �?              �?              �?              �?              �?      �?      �?      �?      @              �?               @      �?              �?              �?               @              @       @      �?       @       @      @      @       @      @      @      @      @      @      �?      @      @      @      &@      $@      "@      "@       @       @      1@      (@      ,@      1@      5@      <@      8@      7@      =@     �@@      C@      C@      C@     �E@      D@     �L@     @R@      J@     �L@     �A@     �J@     �N@      P@      J@     �C@     �D@     �A@     �B@      :@      ,@      @      �?       @        
�
bc1*�	   ��y�   ��+�?      @@!   ��<�?)y}6�I��?2�o��5sz�*QH�x�&b՞
�u�>	� �?����=��?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�������:�              �?      �?              �?              �?       @      �?       @      &@      @      @              �?        
�
bd1*�	   �I�w�    
c�?      P@!   ��}�?)�Ѱ^��?2�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp�����=��?���J�\�?eiS�m�?#�+(�ŉ?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�������:�              ;@       @              @              �?              �?               @       @      @      $@      @       @        
�
bo*�	   `����   ��\�?     �E@!  ��U5Ͽ)Z�<b�?2��uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��-Ա�L�����J�\������=���>	� �����T}�hyO�s�uWy��r�ߤ�(g%k�P}���h�5Ucv0ed?Tw��Nof?�N�W�m?;8�clp?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�#�h/�?���&�?�Rc�ݒ?^�S���?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?�������:�              �?              �?       @      $@      @      �?       @      �?      �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?      �?       @               @      �?      @      �?        �p�w�&      Q*^�	 ��to�A&*�M

loss�(_@

accuracy�~{=
�
wc1*�	    ����   `̑�?      r@!  �!��?)��vg�ߣ?2�}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%�V6��u�w74���VlQ.��7Kaa+��[^:��"��S�F !���Zr[v��I��P=��f�ʜ�7
?>h�'�?�[^:��"?U�4@@�$?��%�V6?uܬ�@8?�!�A?�T���C?a�$��{E?
����G?IcD���L?k�1^�sO?nK���LQ?�lDZrS?ܗ�SsW?��bB�SY?�m9�H�[?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�               @      �?      �?               @      @      @      @       @      @      @      @      @              @      @      @      @       @              @      @      @       @              @      @      @      �?       @      �?              @      �?       @               @      @              �?      �?      �?               @              �?              �?              �?              �?              �?              �?              �?              �?               @       @      �?              �?      �?              �?      �?       @              @       @      �?      @      @      @      @      @      @      $@      (@      $@       @      (@      &@       @       @      *@      @      &@      "@      @      @       @        
�"
wd1*�"	   �꥿   `�ާ?      A!
s==֤�@)�h[�L@2��/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K��R%������39W$:���
�}�����4[_>����
�%W����ӤP���K���7��[#=�؏���i����v��H5�8�t�4�j�6Z�Fixі�W�4�e|�Z#���-�z�!���u}��\>d�V�_>BvŐ�r>�H5�8�t>�����~>[#=�؏�>K���7�>u��6
�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?�������:�              @      @      1@      <@     �H@     ��@     ̞@     ��@     �@     ��@     �@     ȿ@     S�@     ��@    ���@    ���@    ���@     ��@     j�@     �@      �@     ?�@     U�@     �@     ��@     ��@     &�@     ��@     ��@     v�@     ��@     ��@     ��@     Ҥ@     ޢ@     r�@     �@     ��@     �@     Ԗ@     ��@     ��@     ��@     Џ@     P�@     `�@     ��@      �@     8�@     X�@     �@     P@     �y@     �x@      v@      v@     @r@     pq@     �o@      l@      j@     �f@     �c@     �c@     ``@     �^@     �a@     @Y@     @\@      ]@     �V@      O@      I@     �O@      K@      N@     �G@     �H@      @@     �@@      8@      7@      1@      7@      4@      3@      7@      $@      .@      0@      *@      (@      $@       @      @      @       @      @      @      "@      @      @      @       @       @      �?       @      @               @      �?      �?       @              �?              �?               @               @              �?              �?              �?              �?              �?              �?              �?              �?              �?               @              �?      �?              �?      �?              �?              �?              �?              �?      �?      �?       @              �?      �?      @      �?       @       @      �?      @      @      @      @      @      @      @      @      @       @       @      (@      (@      &@      (@      @      2@      0@      .@      0@      0@      7@      ?@      ;@      1@      F@      C@      I@     �B@     �G@      N@     �M@      I@     �T@      P@      [@      V@     �\@     @_@      `@     @a@     `b@     �e@      i@     @g@     `l@     Pq@     @r@     `t@     `u@     �v@     w@     �|@     `@     P�@     ��@     ��@     ��@     Ȉ@     ،@     �@     $�@     �@     p�@     ��@     �@     ��@     Ԝ@     b�@     F�@     ^�@     ��@     �@     ��@     ĭ@     �@     �@     ��@     ��@     ��@     �@     [�@     ��@     ��@     ��@    �;�@     ��@     �@    ���@     /�@    ���@    ���@    �|�@    ��@    �1�@     ��@    �(�@     $�@     �@     ײ@     ��@     0�@     ��@      l@      H@      @        
�
wo*�	    �ߤ�   `�0�?     ��@! T?�O9�)�/��]�?2�	�/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7���x?�x��>h�'��6�]���1��a˲�O�ʗ�����Zr[v����~���>�XQ��>�uE����>�f����>})�l a�>pz�w�7�>��Zr[v�>O�ʗ��>>�?�s��>1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�T7��?�vV�R9?��ڋ?�.�?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?�������:�	               @      @      (@      1@      4@     �C@      F@      F@      O@     �S@     @Q@     @R@     �P@     �R@      N@     �M@     �L@     @P@     �F@     �F@      I@      L@     �H@      @@     �B@      @@      A@      3@      A@      :@      ,@      0@      5@      ,@      (@      *@      (@      "@      $@      ,@      @      @      @      @       @       @      @      �?              @      �?      �?       @      �?      @               @       @      �?      �?      �?      �?               @              �?               @              �?               @              �?              �?              �?              �?      �?               @       @      �?              �?      �?               @              �?              �?       @      �?      @      �?      @       @      @      @              �?      @      @       @      @      @       @      @      &@       @      "@       @      @      @      0@      0@      .@      2@      6@      8@      8@      4@      @@     �B@      B@     �A@      F@     �B@     �C@     �M@      S@     �G@      M@      C@     �H@     @P@     @P@      J@      B@      E@      ?@     �D@      ;@      (@      @      �?       @        
�
bc1*�	    ��y�   ��s�?      @@!  ���~�?)!cS�?2�o��5sz�*QH�x�&b՞
�u�����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�������:�              �?      �?              �?              �?      �?       @       @      &@      @      @              �?        
�
bd1*�	    x�   @ę?      P@!  ��c��?)�S�|�?2�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp�����=��?���J�\�?eiS�m�?#�+(�ŉ?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�������:�              ;@       @      �?      @              �?              �?               @       @      @      "@      @      @        
�
bo*�	   `���   ��ס?     �E@!  ��w�Ͽ)������?2��uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��-Ա�L�����J�\������=���>	� �����T}�&b՞
�u�hyO�s�;8�clp��N�W�m�Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?�������:�              �?      �?              @      "@      @      �?       @      �?      �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?      �?      �?      �?              �?       @      @      �?        ���>&      Sd�	Zxuo�A'*�L

lossU,_@

accuracy��}=
�
wc1*�	   �치�   �P�?      r@!  ᤝ��?)��b s��?2�}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8��u�w74���82�I�I�)�(�+A�F�&��[^:��"��S�F !�ji6�9��I�I�)�(?�7Kaa+?uܬ�@8?��%>��:?d�\D�X=?���#@?�T���C?a�$��{E?
����G?�qU���I?IcD���L?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�               @      �?      �?               @      @      @      @      @      @      @      @      @              @      @      @       @       @       @      @      @      @              @       @      @       @              �?      @      �?              @      @              �?      �?              @              �?              �?              �?              �?              �?      �?              �?              �?              �?              �?               @      �?              �?              �?              �?              @              @       @      �?      @       @      @      @       @      @      @      "@      (@      @      $@      *@      "@      "@      @      (@       @      &@      @      @      @       @        
�!
wd1*�!	    �)��   ��B�?      A!Y�#Щ��@)����kL@2��/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�������������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:�����ӤP�����z!�?��T�L<��[#=�؏�������~�=�.^ol�ڿ�ɓ�i�w&���qa�d�V�_���Ő�;F��`�}6D�K���7�>u��6
�>���m!#�>�4[_>��>
�}���>X$�z�>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?�������:�              @      @      3@      A@     �K@     X�@     <�@     ��@     W�@     Ǹ@     T�@     �@    ���@    �#�@    ���@     ��@     �@    ���@    ���@    ��@     8�@     ��@     y�@     ��@     ��@     �@     �@     �@     ��@     ��@     0�@     �@     ��@     ��@     ̢@     ~�@     ̟@     ��@     x�@     T�@     ��@     ��@     |�@     ,�@     P�@     ��@     ��@     Є@     ��@      �@     �@     P~@      {@     �x@     �x@     `u@      s@      p@     �p@     �l@     �j@     �f@     �c@     `d@     �b@      `@     �]@     �[@      [@     �[@      S@      L@     �N@     �P@     �F@      H@     �B@      G@      A@      ?@      8@     �@@      4@      ,@      5@      2@      5@       @      ,@      (@      ,@      (@      @      @      @      @      $@      @      @      @      @      @      @      @       @               @      @      �?              �?               @              @       @       @              @              �?              �?       @               @              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?              �?      �?      �?               @      �?      �?      �?      @      @      �?      @      @      @      @       @      @       @      "@      "@      @      $@      1@      $@      @      "@      *@      1@      ,@      3@      4@      <@      ?@      :@      @@      E@     �D@      E@      D@      G@     �M@     �M@     �P@     �T@      V@     �V@     @Z@     �Z@     @^@      ^@      a@     `f@     �f@      f@     �h@     �j@     `q@     �s@     �s@     Pv@     �u@     @z@     �}@      ~@      �@     0�@     H�@     0�@     ��@     ��@     ��@     �@     �@     d�@     `�@     @�@     \�@     ��@     �@     4�@     ��@     ��@     �@     ܩ@     �@     �@     �@     {�@     ޴@     ��@     ��@     ��@     ��@     8�@    ���@    �?�@     s�@    ��@     ��@    �)�@    ��@     ~�@     ��@     ��@    ���@     r�@    ���@     ��@     ��@     v�@     D�@     �@     ��@      n@      L@      $@        
�
wo*�	   ��٤�   ಊ�?     ��@! �i2�
�)�Ⱥ��J�?2�	�/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"�ji6�9���.���vV�R9��T7���f�ʜ�7
������1��a˲���[���FF�G ����%ᾙѩ�-߾['�?�;;�"�qʾ['�?��>K+�E���>��~]�[�>��>M|K�>�iD*L��>E��a�W�>��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�������:�	               @      @      ,@      .@      2@      E@      E@     �F@      P@      S@     @Q@     �Q@      Q@     @S@      M@      P@     �J@     �N@     �I@      D@     �K@     �J@      I@      A@     �C@      >@      ?@      6@      A@      9@      ,@      0@      2@      ,@      ,@      1@       @      (@      @      &@      @      @      @       @      @      �?       @      @      @      @       @       @      �?      @               @      @       @       @      �?              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?       @      �?      �?      �?      �?              �?       @      �?      �?      �?              �?      �?      �?               @      @      @      �?      �?      @       @      �?      �?      @      @      @      @      �?       @      @      &@      "@      &@      @      @       @      (@      1@      1@      0@      5@      =@      4@      6@      >@     �@@      C@     �C@      D@     �C@      C@     �M@     �R@     �J@     �K@      D@     �G@      P@     @P@      J@      B@      F@      >@      D@      <@      (@      @       @      �?      �?        
�
bc1*�	   @��y�   `ޗ?      @@!   ���?)�N�T�2�?2�o��5sz�*QH�x�&b՞
�u�����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�������:�              �?      �?              �?              �?      �?       @       @      $@      @      @              �?        
�
bd1*�	   ��x�    7+�?      P@!  �����?)>�&p�?2�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp�����=��?���J�\�?eiS�m�?#�+(�ŉ?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�������:�              1@      $@       @      �?      @              �?              �?               @       @      @      @       @      @        
�
bo*�	   �kv��   �^O�?     �E@!  @Ӓ)п)�=��-��?2�`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY�����J�\������=���>	� ��o��5sz�*QH�x�uWy��r�;8�clp�Tw��Nof?P}���h?ߤ�(g%k?����=��?���J�\�?eiS�m�?#�+(�ŉ?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?��<�A��?�v��ab�?�/��?�uS��a�?�������:�              �?              �?              @      $@      @               @      �?      �?               @      �?              �?              �?              �?      �?              �?              �?              �?      �?      �?      �?              @       @       @        6�}�%      [0$�	Q�2vo�A(*�K

loss�_@

accuracyڬz=
�
wc1*�	   @Po��   ���?      r@!  F9��?)��shcC�?2���<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@���%>��:�uܬ�@8���d�r?�5�i}1?I�I�)�(?�7Kaa+?��bȬ�0?��82?��%>��:?d�\D�X=?���#@?�!�A?a�$��{E?
����G?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?      �?      �?      �?               @       @      @      @      @      @      @      @      @      @      �?      @      @      �?       @      @      @       @       @      @      @       @       @              @      �?      �?      @      @      �?       @      �?      �?      �?      �?      �?      �?               @              �?              �?              �?              �?               @              �?              �?              �?              �?               @      �?       @       @      @      @      @      @      @       @      @       @      @      &@      &@      &@       @      (@      @      $@      @      $@      $@      *@      @      @      @      �?        
�!
wd1*�!	    cm��   ����?      A!�h�.��@)��п]fL@2��g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:����4[_>������m!#��ہkVl�p>BvŐ�r>�H5�8�t>�i����v>u��6
�>T�L<�>���m!#�>�4[_>��>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�              �?      @      �?      7@      B@      N@     ��@     `�@     ��@     ��@     ��@     ��@      �@    ���@    �6�@    ���@     ��@     4�@     ��@    ���@     +�@     >�@     ��@     ��@     4�@     x�@     =�@     �@     �@     ��@     ��@     ��@     R�@     `�@     ��@     ��@     X�@     (�@     ��@     �@     �@      �@     x�@     \�@     x�@     P�@      �@     ��@     @�@     `�@     ��@     (�@     P~@     P{@     `y@     0w@     v@     @s@      r@      p@     `n@     �i@     �d@     �e@     �f@      b@     �a@      ^@     @Y@     �^@     �U@     @U@     �M@     �K@     �M@     �K@      J@     �A@      F@     �@@      @@      ;@      <@      >@      <@      4@      5@      2@      "@      *@      "@       @      *@      "@       @      $@      @      @      @       @      @      @      �?       @      @      @      @      @      @      �?      �?      @       @      @      �?      �?       @      �?      �?               @      �?       @              �?              �?              �?              �?              �?              �?              �?      �?              �?      @      �?      �?       @       @      �?       @       @      �?       @       @      @      �?      @      @      @      @      @      @      @      @      "@      1@      &@      *@       @      5@      &@      ,@      ,@      2@      4@      7@      <@      <@      ;@      =@      :@     �C@      @@      >@     �H@     �J@     �R@     �P@      R@      Q@      W@      Y@     @[@     @[@     ``@      a@     �f@     @e@     �i@     �f@     @m@     @p@     �r@      s@     �v@     �u@     z@     �z@     �}@     �@     ��@     ��@     ��@     `�@      �@     �@     �@     ��@     �@     ��@     �@     �@     d�@     �@     �@     ��@     ��@     b�@     �@     |�@     ��@     �@     �@     ��@     5�@     ʸ@     Z�@     ��@      �@    ���@    �:�@     i�@    ���@     ��@    �,�@    ���@     W�@     ��@     d�@    ���@     @�@     ��@     �@     p�@     A�@     ��@     �@      �@     @p@     �O@      (@      �?        
�
wo*�	   �դ�   ���?     ��@! b����)w��Z@�?2�	�/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��S�F !�ji6�9����ڋ��vV�R9�x?�x��>h�'��f�ʜ�7
�I��P=��pz�w�7���ߊ4F��h���`ѩ�-߾E��a�Wܾ��~]�[�>��>M|K�>�f����>��(���>I��P=�>��Zr[v�>>�?�s��>�FF�G ?��[�?1��a˲?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�������:�	               @      @      .@      .@      3@     �C@     �E@     �E@      Q@      S@     �P@      Q@     @S@     @R@     �L@     �P@      J@     �M@     �G@     �F@      J@      M@      J@      ?@     �B@      B@      ;@      7@      A@      6@      2@      *@      6@      ,@      (@      *@      @      *@      @      .@      @      @      @      @      @      @       @      �?      @      @      �?      @       @      �?              @      @       @      �?               @              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?       @      �?      @               @              �?               @      �?      @       @      �?       @       @      @      �?              @      @      @       @      @       @      @       @      $@      &@      @      @      @      (@      0@      3@      3@      4@      8@      ;@      5@      <@     �@@      B@      C@     �B@     �E@     �F@     �K@     @S@      I@     �K@     �C@     �H@      O@     �P@     �I@      C@      E@     �@@     �C@      ;@      (@      @       @               @        
�
bc1*�	   �;�y�   @�v�?      @@!   pg$�?)�&�e��?2xo��5sz�*QH�x�&b՞
�u����J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�������:x              �?      �?              �?      �?      �?       @      �?      "@      @       @              �?        
�
bd1*�	   @�x�   �G��?      P@!  ��d6�?)�t����?2�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp�����=��?���J�\�?eiS�m�?#�+(�ŉ?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�������:�              5@      @       @      �?      @              �?              �?               @      @      @      @       @      @        
�
bo*�	   `OѢ�   `�¢?     �E@!  �ssп)٤L�g�?2�`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY�����J�\������=���>	� �����T}�o��5sz�&b՞
�u�hyO�s�5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?����=��?���J�\�?-Ա�L�?eiS�m�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�������:�              �?              �?      �?      @      $@      @               @      �?      �?               @      �?              �?              �?              �?              �?              �?              �?              �?      �?      �?      �?              @      �?       @      �?        �0><%      X�	$�\wo�A)*�J

lossF_@

accuracyn4�=
�
wc1*�	   @?X��   ���?      r@!  2&?��?)t^q�?2���<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@��ߊ4F��h���`��5�i}1?�T7��?ji6�9�?�S�F !?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?�!�A?�T���C?
����G?�qU���I?nK���LQ?�lDZrS?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?      �?      �?      �?              @      @      @      @      @      @      @       @      @      @      �?      @      @       @      @      @      @      @       @      @       @      @      �?      @      �?              @       @      �?       @       @      �?       @              �?      �?      �?              �?              �?              �?              �?              �?      �?              �?              �?              �?              �?               @              @       @       @      �?      @       @      @      @      @      @      @      "@      "@      *@       @      $@      $@       @      "@      "@       @      (@      (@      @      @      @      �?        
�!
wd1*�!	   @崦�    ��?      A!H����d�@)���5wL@2��g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���
�}�����4[_>���T�L<��u��6
��BvŐ�r>�H5�8�t>�����~>[#=�؏�>��ӤP��>�
�%W�>�4[_>��>
�}���>X$�z�>.��fc��>R%�����>�u��gr�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�              �?      @      @      ;@     �D@     �P@     h�@     ؟@     D�@     ��@     �@     ��@    �E�@     ��@     T�@    ���@     ��@    �*�@    ���@     ��@    �0�@     �@     Ѽ@     ��@     �@     ��@     �@     .�@     P�@     ��@     *�@     �@     ʨ@     ��@     �@     ¢@     H�@     8�@     ��@     t�@     H�@     �@     4�@     <�@      �@      �@     ��@     0�@     0�@     Ȅ@     ��@     X�@     �}@     @{@     Py@     pv@     �v@      s@     Pq@     @o@     �n@      g@      e@     �d@     �c@     �c@     �]@     �\@     @]@     �[@      V@     �Q@     �J@      Q@     �N@      H@      C@      E@      H@      E@      B@      :@      9@     �B@      3@      8@      3@      2@      (@      (@      3@      *@      &@      *@      @      &@      @      @      @      @       @       @      @      �?      @       @      @      @       @      �?      @      �?       @      �?      �?              �?      �?       @              �?              �?              �?              �?              �?              �?               @              �?              @              �?               @       @      �?      �?      �?      @              �?      @      @              @      @      @      @      @      @      @      @      @       @      "@      @      @      @      ,@      $@      "@      $@      &@      0@      *@      *@      7@      ;@      8@      8@      >@      B@      I@      A@     �G@     �N@      L@     �N@     �T@     �R@      T@      Y@     @X@     �Y@     @[@     `a@      `@      f@     �f@     @i@     �h@      i@     �p@     �r@     `t@     0u@     @v@     Py@     �{@     �|@     ��@     �@      �@     ؆@     `�@     �@     @�@     H�@     ��@     ē@     t�@     x�@     4�@     �@     0�@     <�@     ��@     �@     |�@     4�@     b�@     ��@     ,�@     �@     ��@      �@     t�@     ��@     м@     �@    ���@     �@    �p�@     ��@    ���@    �3�@     ��@     <�@     _�@     �@    ���@     ��@     ��@     �@     d�@      �@     �@     P�@     ��@      r@     @S@      2@       @        
�
wo*�	    Ϥ�   �O^�?     ��@! ����)ғ��UB�?2�	�/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&��[^:��"��S�F !�ji6�9���.����ڋ�x?�x��>h�'��O�ʗ�����Zr[v��})�l a��ߊ4F���_�T�l�>�iD*L��>a�Ϭ(�>8K�ߝ�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�������:�	               @      @      ,@      0@      1@     �D@      F@     �D@     @Q@     �R@     �P@     �Q@     �S@      Q@      N@      Q@      H@     @P@      E@     �G@     �I@     �N@     �H@      :@     �G@      @@      <@      5@     �A@      6@      0@      4@      4@      ,@      $@      $@      $@      $@      @       @      $@      $@      @      @      @      @       @      @      @      @       @      @      �?      �?      �?      @      @       @               @               @       @              �?              �?              �?               @              �?              �?              �?              �?      �?       @      �?       @      �?               @      �?      �?      @      @               @       @       @      �?      �?      @      $@      @      �?      @      @      $@       @      $@      "@      @      @      @      *@      3@      ,@      3@      9@      <@      6@      1@      ?@     �@@      B@     �@@     �D@     �E@     �G@     �L@     �R@      H@      K@     �D@      I@     �K@     @Q@      I@     �E@     �C@      B@      B@      ;@      ,@      @       @               @        
�
bc1*�	   `}�y�   ��;�?      @@!  ��B��?)�UN�]�?2xo��5sz�*QH�x�&b՞
�u����J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?}Y�4j�?��<�A��?�������:x              �?      �?              �?               @       @      �?       @      @      "@              �?        
�
bd1*�	   ��(x�   ���?      P@!   R���?)�����l�?2�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp�>	� �?����=��?eiS�m�?#�+(�ŉ?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�������:�              5@      @       @       @      @              �?              �?               @       @      @      @       @      @        
�
bo*�	   �C-��    G2�?     �E@!  @0��п)MnƔ��?2�`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/�����J�\������=���>	� �����T}�*QH�x�&b՞
�u��l�P�`?���%��b?P}���h?ߤ�(g%k?����=��?���J�\�?-Ա�L�?eiS�m�?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�������:�              �?              �?      �?      @      $@       @               @       @               @      �?      �?              �?              �?              �?              �?              �?              �?              �?      �?      �?               @       @       @      �?        ���%      D*�_	nlxo�A**�I

loss�_@

accuracyI��=
�
wc1*�	   ����    �Λ?      r@! �B�u�?)]�0��?2���<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C���d�r�x?�x���
�%W����ӤP����FF�G ?��[�?�S�F !?�[^:��"?I�I�)�(?�7Kaa+?d�\D�X=?���#@?�T���C?a�$��{E?nK���LQ?�lDZrS?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?      �?      �?      �?      �?       @      @      @      @      @      @      @      @      @      @       @      @      @      �?      @      @       @       @      @      @       @       @      �?      @              @       @              @       @       @       @      �?               @              �?              �?              �?               @              �?              �?              �?              �?              �?               @              @      �?       @      @      @      @      @      @       @      @      @       @      "@      (@      $@      "@      "@      "@       @      "@       @      *@      (@      @      @      @      �?        
� 
wd1*� 	     ��    ���?      A!}7�,�@)�<dC�L@2��g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g����u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]�����MZ��K���u��gr��R%������39W$:���T�L<��u��6
��BvŐ�r�ہkVl�p�H��'ϱS��
L�v�Q�u��6
�>T�L<�>�
�%W�>���m!#�>39W$:��>R%�����>�u��gr�>�MZ��K�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�              �?      @      @      A@     �E@     �R@     P�@     �@     ��@     �@     K�@     ٽ@    �T�@     ��@     ��@    ���@    ���@    �4�@     ��@    ���@     �@     &�@     Լ@     ú@     �@     t�@     �@     ��@     ^�@     l�@     *�@      �@     0�@     .�@     ֤@     �@     ��@     ��@     h�@     ��@     ��@     8�@     �@     ��@     P�@     �@     x�@     ��@     ��@     �@     ��@     (�@     �}@     �y@     �z@     pw@      u@     �q@     0r@     `n@     �l@      i@     �f@      d@     �f@      c@      `@     �_@      [@     @\@     �S@     �R@     �P@     �L@     �J@     �H@     �F@     �A@     �E@      @@      C@     �B@      :@      =@      0@      6@      3@      &@      (@      0@      .@      (@      .@      0@      @      @      @      @      $@      @       @      @      �?      @      @       @      @       @      @              �?      �?      @              �?               @              �?               @              �?              �?              �?              �?               @              �?      �?      �?              �?               @       @      �?              @              @       @      �?       @      @      @       @      @      @      @      @      @      @      "@      "@      "@      ,@      @      0@      ,@      1@      0@      .@      2@      8@      1@      7@      :@     �D@      I@     �C@      @@      C@      G@      K@      O@     �M@     �V@     �Q@      U@     @Z@     �^@     @[@     �\@     @b@     �g@     �e@     �i@     `g@     �j@     q@     Pr@      t@      u@     �w@     �y@     |@     �}@     �@     �@     ��@     h�@     �@     ؋@     ��@     ܐ@     ��@     �@     D�@     ��@      �@     h�@     �@     *�@     ��@     |�@     ��@     Ω@     `�@     ��@     P�@     ��@     ��@     9�@     @�@     ��@     ɼ@     3�@    ���@    ��@    �J�@     ��@     ��@     �@     ��@    ��@    �:�@    ��@    ���@     ��@     ��@      �@     l�@     /�@     4�@     ��@     P�@     �t@     @W@      9@       @        
�
wo*�	   @�Ƥ�   `�У?     ��@! �����)ƕ� /Q�?2�	�/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��.����ڋ��vV�R9���d�r�x?�x��1��a˲���[��>�?�s���O�ʗ���I��P=��pz�w�7��E��a�W�>�ѩ�-�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�������:�	               @      @      *@      ,@      3@     �D@      G@      E@     @Q@     @R@      O@     @S@      S@     �Q@     �N@     �P@      F@     �P@     �D@     �H@      J@      N@      H@      ;@      G@      A@      =@      2@      B@      9@      .@      1@      6@      &@      *@      "@       @      $@      @      @      *@      @      @      @      @      @      @       @      @      @      �?      @       @      �?      �?      @      @      �?      �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @      �?       @       @              �?               @      @               @      @              �?       @      @              @      @      @      @      �?      @      @      &@       @       @      "@      @      �?      $@      0@      ,@      2@      3@      6@      >@      6@      0@     �@@     �@@     �@@      A@      F@      E@      E@     �M@     �Q@      I@      L@      F@     �F@     �K@     �P@     �I@     �G@      B@     �B@      B@      :@      ,@      @       @               @        
�
bc1*�	   ��z�   �N&�?      @@!  �@q�?)�U|����?2po��5sz�*QH�x�&b՞
�u�-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?}Y�4j�?��<�A��?�������:p              �?      �?              �?       @      �?      �?      @      $@      "@              �?        
�
bd1*�	    $3x�   ����?      P@!   j?��?)D��9�	�?2�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp�>	� �?����=��?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�������:�              5@      @       @       @      @              �?              �?              �?      �?       @      @      @      @      @        
�
bo*�	    ���   @U��?     �E@!  �&7ѿ)�	�\ԗ?2�`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x���bB�SY?�m9�H�[?ߤ�(g%k?�N�W�m?����=��?���J�\�?-Ա�L�?eiS�m�?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�������:�              �?              �?       @      @       @       @               @       @              �?      �?      �?      �?              �?              �?              �?              �?              �?              �?              �?      �?      �?               @      �?      @      �?        $���<&      bH[2	�;�xo�A+*�L

loss�_@

accuracy�ew=
�
wc1*�	   �����   `���?      r@! @ȿ�'�?)B�F��?2��v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��T7����5�i}1���[���FF�G ��f�����uE����O�ʗ��>>�?�s��>x?�x�?��d�r?ji6�9�?�S�F !?��%>��:?d�\D�X=?���#@?�!�A?k�1^�sO?nK���LQ?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?              �?      �?      �?       @      �?      @      @      @      @      @      @       @      @      @      @      @      @      �?      @      @       @       @      @      @      @              @      �?      �?      @      �?       @       @      �?      @      �?               @      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @              �?      @      �?      �?      @      @      @      @      @       @      @      @      &@      @      &@      $@      "@      $@       @       @      "@       @      *@      (@      @      @      @      �?        
�!
wd1*�!	    pQ��   @���?      A!���m�;�@){�\�o�L@2��g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W���i����v��H5�8�t���ӤP��>�
�%W�>���m!#�>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�              @      @      "@      B@     �F@     �V@     ��@     2�@     ƨ@     ��@     q�@     ��@     v�@    ���@     ��@     ��@     ��@     F�@    ���@     ��@     ��@     5�@     ��@     ��@     �@     ��@     )�@     |�@     '�@     �@     �@     T�@     ƨ@     �@     (�@     n�@     �@     \�@     H�@     Ԛ@     �@     Ԗ@     $�@     ��@     А@     ��@     ��@     Ȉ@     ��@     ��@     �@     �@     P}@     �{@     z@     px@      v@     @r@     @q@     Pp@     �m@     `j@      f@      g@     `f@     �a@     @\@      \@      [@     �V@     �W@     �R@     �P@     �M@      G@     �H@      H@      E@     �F@      H@     �D@      7@      :@      <@      2@      1@      2@      .@      1@      0@      @      *@      &@      @      $@      @      @      @      @      @       @      @       @       @      �?      @       @      @      @      �?      �?      �?              @      @              �?      �?              �?               @      �?      �?              �?      �?      �?      �?              �?              �?      �?              �?      �?      �?      �?      �?      @      �?      �?      �?      @              �?      @      �?       @      �?      �?       @              @       @      @      @      @      @      @      @      @      @      @      "@      ,@      @      0@      .@      (@      &@      0@      7@      ;@      2@      9@      >@      9@      D@      F@      D@     �@@     �B@     �J@      G@      J@     @P@     @R@     @S@     �T@     @Y@     �\@      _@     @]@      d@     �e@      f@     `g@     �h@     �j@     @p@     �s@     �t@     �u@     0v@     @x@     �y@     �}@     8�@     �@     P�@     ȅ@     ��@     ��@     ��@     �@     �@     ��@     ��@     ��@     @�@     ��@     f�@     >�@     @�@      �@     ��@     8�@     R�@     8�@     A�@     ��@     ´@     R�@     �@     e�@     ��@     @�@    ���@    ���@     @�@    ���@    ���@     ��@    �v�@     ��@     &�@     �@    ���@     ��@    ���@     ;�@     \�@     ��@     ��@     ��@     L�@     �w@      ^@      >@      @        
�
wo*�	   �����   @kN�?     ��@! �ۅ��)�m��m�?2�	�/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.���vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���8K�ߝ�a�Ϭ(龙ѩ�-�>���%�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�������:�	               @      @      ,@      *@      3@     �C@      G@      F@     �Q@     @R@      O@     @T@      R@      R@     @P@      O@     �F@      P@     �D@     �I@      L@      L@     �H@      =@      F@      ?@     �@@      2@      C@      7@      (@      2@      7@      &@      $@      &@       @      (@       @      @      &@      @       @      �?      &@      @      @       @      @      @       @      @      �?               @      @      �?      �?              �?       @      �?      �?              �?               @              �?      �?              �?              �?              �?              �?              �?              �?              �?               @      �?      �?              �?      �?      �?      �?       @       @              �?               @      �?      �?      �?      @      @               @      @      @      @      @      @      @      @      @      (@      @       @      @      @      @      @      *@      0@      5@      2@      8@      >@      2@      4@      <@      A@      B@      =@     �G@     �D@      F@     �N@      P@     �K@     �J@     �F@     �F@      K@     �P@     �I@     �G@      B@      C@     �@@      =@      ,@      @      @              �?      �?        
�
bc1*�	   ��z�   ��/�?      @@!  @��`�?)���j�e�?2po��5sz�*QH�x�&b՞
�u�-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?}Y�4j�?��<�A��?�������:p              �?      �?              �?       @      �?      �?      @      &@      "@              �?        
�
bd1*�	   @b<x�   ���?      P@!   
X[�?){7D��?2�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp�>	� �?����=��?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              5@      @       @       @      @              �?              �?              �?      �?              @      @       @      @       @        
�
bo*�	    �ꣿ   `c�?     �E@!  �p�`ѿ)_38ݓ�?2�`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/��-Ա�L�����J�\������=������T}�o��5sz�nK���LQ?�lDZrS?�N�W�m?;8�clp?>	� �?����=��?���J�\�?-Ա�L�?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�������:�              �?      �?      �?       @      @      @       @               @       @              �?       @               @              �?              �?              �?              �?              �?              �?      �?      �?              �?       @       @       @        ;�1v�&      `ɠ	9��yo�A,*�M

loss��^@

accuracyn4�=
�
wc1*�	   `�;��   ����?      r@! ����?)�NCg��?2��v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C���%>��:�uܬ�@8�>h�'��f�ʜ�7
�6�]���1��a˲�pz�w�7��})�l a�;�"�qʾ
�/eq
Ⱦ����?f�ʜ�7
?�vV�R9?��ڋ?d�\D�X=?���#@?k�1^�sO?nK���LQ?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?              �?      �?              @      �?      @      @      @      @      @      @      �?      @      @      @      @      @      �?      @      @      @       @      @      @      @              @      �?       @      @               @       @      �?      @      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?               @              �?              �?      �?      �?              �?      @      �?      �?      @      @      @      @      @      @       @      @      &@      @      (@      "@      "@      $@       @       @      "@       @      ,@      &@      @      @      @      �?        
�"
wd1*�"	    ڪ��   �pk�?      A!#yۅV��@)R��b'M@2��g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%�������4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��f^��`{�E'�/��x�=�.^ol�ڿ�ɓ�i�4�j�6Z�Fixі�W�����W_>�p
T~�;������0c>cR�k�e>:�AC)8g>[#=�؏�>K���7�>u��6
�>��z!�?�>��ӤP��>���m!#�>�4[_>��>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�              @      @      1@     �C@     �H@     �[@     Ѓ@     n�@     ��@     �@     ��@     %�@     ��@     ��@     ��@    ���@    ���@     V�@    ���@    ���@    ��@     D�@     ��@     º@     θ@     ��@     $�@     n�@     �@     N�@     \�@     �@     ب@     
�@     \�@     ΢@     �@     �@     �@     �@     ��@     L�@     Ȕ@     �@     ��@     Ѝ@     ��@     (�@     �@     ��@      �@     ��@     0}@     �z@     @x@     �w@     �t@     `s@     Pr@     �n@     �l@     �i@     �h@     `f@      b@     �a@     @]@     �[@     @Z@     �X@     @Y@     �U@     �K@      I@      I@      M@     �H@      E@     �F@     �H@      E@      ?@      7@      3@      4@      4@      0@      ,@      *@      .@      $@      .@      (@      &@      "@      @      "@      @      @       @      @      @      @      @      �?      @      @      @      @       @      �?      @      �?               @      �?      �?      �?      �?              �?      �?              �?              �?      �?              �?       @               @              �?              �?              �?              �?      �?              �?      �?               @              �?              �?      �?      @      �?              �?              �?      @       @      �?      �?       @       @      @      @      @      @      @       @      $@      (@      @      (@      &@      7@      2@      .@      1@      3@      >@      1@      3@      7@      @@      9@      A@      @@     �C@      C@     �E@     �G@      D@      D@     �P@     �S@     @W@     �S@      W@     �Z@     �\@      a@      b@      f@     `f@      g@      i@      k@     �p@     �s@     �q@     v@     �v@     �x@     �y@     0|@     Ѐ@     0�@     ��@     h�@     ��@     @�@      �@     ,�@     \�@     \�@     �@     �@     (�@     �@     �@     �@     ��@     0�@     ��@     ��@     �@     �@     �@     ��@     ��@     j�@     �@     �@     ��@     q�@     }�@     ��@    �#�@     ��@    �{�@    ���@     ��@     ��@    ��@     �@     ��@     ��@    ���@     7�@     ��@     
�@     N�@     d�@     p�@     �{@     `a@      F@      @        
�
wo*�	   �����   @<Ҥ?     ��@! �)7�)��μ��?2�	�/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�+A�F�&�U�4@@�$�ji6�9���.���T7����5�i}1���d�r�>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s�����Zr[v��I��P=��8K�ߝ�a�Ϭ(龮��%�>�uE����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>})�l a�>pz�w�7�>I��P=�>6�]��?����?f�ʜ�7
?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�������:�	               @      @      ,@      (@      7@     �B@      F@     �G@      Q@     @R@     @P@     @T@     @R@     @R@     �N@      O@      J@     �O@     �D@     �G@      K@      M@      H@      @@     �F@      <@      A@      6@      @@      5@      2@      2@      4@      (@      (@      $@      @      *@      "@      @      (@      @      @      @      @      @       @      @      @       @       @      @               @      @       @               @               @              �?      �?              �?              �?               @              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?              �?      �?       @              �?       @      �?              �?       @      �?      @      �?      �?      @      �?      �?      @      @      @       @      @      @      @      @      @      $@      @       @       @      @       @      @      1@      0@      0@      5@      5@      <@      3@      6@      >@      <@      C@     �@@      G@     �A@      K@     �M@      O@     �H@      N@     �C@      H@     �J@     @P@      J@     �G@     �A@     �C@      ?@      ?@      ,@      @       @      �?               @        
�
bc1*�	   @z�    �Q�?      @@!  @X��?)1R��8�?2po��5sz�*QH�x�&b՞
�u�eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?��<�A��?�v��ab�?�������:p              �?      �?               @       @      �?      @      $@      $@      �?              �?        
�
bd1*�	   ��Dx�   �)��?      P@!  ���?)H�=�?2�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp�>	� �?����=��?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              6@      @       @       @      @              �?              �?              �?      �?              @      @      @      @      @        
�
bo*�	    M��    �g�?     �E@!  ��j�ѿ)�#l�X�?2��/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/��eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz��!�A?�T���C?;8�clp?uWy��r?>	� �?����=��?���J�\�?-Ա�L�?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�������:�              �?              �?      �?      @      @      @       @      �?      �?       @              �?      �?      �?              �?      �?              �?              �?              �?              �?              �?              �?      �?      �?              �?       @       @      �?      �?        �
�+|&      cT?�	�w4zo�A-*�L

lossF�^@

accuracyJ{�=
�
wc1*�	   �ʞ��   @X��?      r@! �xt��?)�Α�
�?2��v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L�
����G�a�$��{E��T���C��T7����5�i}1�>h�'��f�ʜ�7
�E��a�Wܾ�iD*L�پO�ʗ��>>�?�s��>��[�?1��a˲?x?�x�?��d�r?�vV�R9?��ڋ?d�\D�X=?���#@?�!�A?k�1^�sO?nK���LQ?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?              �?      �?              @      �?      @      @      @      @      @      @      �?      @      @       @      @      @      �?      @      @       @      @      @      @      @              @      �?       @      @      �?       @       @       @       @               @      �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?               @               @       @       @      �?      @       @      �?      @      @      @       @      @      &@      @      (@       @      $@      $@       @       @      "@       @      ,@      &@      @      @      @      �?        
�"
wd1*�"	   � ��   @���?      A!HnB�s	�@)���|�M@2��g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R�����]������|�~���MZ��K��R%������39W$:���.��fc����4[_>������m!#���
�%W����ӤP���BvŐ�r�ہkVl�p���u}��\�4�j�6Z�w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>K���7�>u��6
�>��z!�?�>��ӤP��>�4[_>��>
�}���>X$�z�>.��fc��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?�������:�              @      @      8@      C@     @P@      \@     @�@     ��@     :�@     �@     ��@     5�@    ���@    ���@    ���@    ���@    ���@    �E�@    ���@     ��@    ��@     (�@     ��@     Ӻ@     �@     ��@     (�@     !�@     �@     v�@     �@     ��@     ��@     z�@     ��@     ��@      �@     ԟ@     �@     �@     d�@     �@     ��@     x�@     Ԑ@     �@     ��@      �@     �@     ��@     8�@      �@      ~@      {@     �w@     �v@     �t@     �s@     �q@     @n@     �l@     �h@     �g@     �h@     �c@      _@     ``@     �Z@     �X@     �^@     �U@     @Q@     �K@     �L@     �N@     �L@      D@      L@      F@      C@     �@@      A@      .@      :@      5@      1@      @      3@      ,@      0@      0@      (@      &@      "@       @      "@       @      @      $@      @      @       @       @      @       @      @      @      �?      @      �?       @       @       @      �?              @              �?      �?              �?      �?              �?               @              �?              �?              �?              �?              �?              �?              �?              �?              �?              @       @              @       @      �?       @      �?       @               @      @      �?      �?      @      @      @       @      �?       @      @       @      "@      @      0@       @      ,@      0@      "@      (@      ,@      ,@      4@      3@      3@      0@      0@      8@      4@      2@      >@     �A@     �E@      C@      D@     �I@      B@      N@     �M@     �Q@     @R@     �V@     �Y@      V@     �`@     �a@     �`@     `e@     �f@      g@     �h@     �m@     q@     �q@     �r@     `u@      v@      y@     �y@     �~@     8�@     ��@     h�@     x�@     ��@     ȋ@     (�@     $�@     ��@     ��@     ĕ@     L�@     �@     ��@     ��@     �@     ��@     8�@     ��@     ��@     �@     D�@     ̰@     ��@     s�@     w�@     .�@     ��@     ��@     9�@     ��@    ���@     ��@     ��@     q�@     ��@     ��@    ���@    �>�@     ��@    ���@    ���@    ���@     ��@     ٷ@     g�@     v�@     ��@     ��@     ��@     �d@     �J@      0@      �?        
�
wo*�	    a���   ��X�?     ��@!  lU�r�)��p{��?2�	�/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$�ji6�9���.���vV�R9��T7����5�i}1�x?�x��>h�'��f�ʜ�7
������6�]���I��P=��pz�w�7���_�T�l׾��>M|Kվ�[�=�k���*��ڽ��ѩ�-�>���%�>�uE����>�f����>��(���>��[�?1��a˲?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�������:�	               @      @      ,@      &@      ;@      A@     �G@      H@     �P@     @R@     �P@     @T@     �R@     �Q@      M@     @P@     �J@     �N@     �E@     �H@     �I@      L@     �I@      =@      H@      :@      C@      2@      ?@      6@      6@      4@      0@      $@      &@      $@      "@      .@      @      "@      *@      @       @      @      @      @              @      @      @      @      @               @      @      @      �?               @              �?              �?      �?              �?      �?              �?              �?              �?              �?              �?              �?      �?              �?              �?      �?       @       @       @      �?               @      �?      �?              @      @      �?      �?      �?      @      @      �?      @      @      �?      @      @      $@       @      @      @      $@      @      $@      0@      *@      4@      ,@      5@      :@      5@      7@      ?@      <@     �@@     �C@      F@     �@@      K@      K@     �P@     �H@      L@      E@     �G@      K@     �M@     �N@      D@     �D@      D@      =@      <@      2@      @       @      �?               @        
�
bc1*�	    l z�    ��?      @@!  �>��?)���z!�?2ho��5sz�*QH�x�&b՞
�u�#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?��<�A��?�v��ab�?�������:h              �?      �?              @       @       @      $@      @      @              �?        
�
bd1*�	    qLx�   ��.�?      P@!   7Ya�?)/]ONg-�?2�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp�>	� �?����=��?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              7@      @       @       @      @              �?              �?              �?      �?              @      @      @      @      @        
�
bo*�	   �P���   ��Ȥ?     �E@!  p�<ҿ)����!�?2��/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� ��o��5sz�*QH�x��vV�R9?��ڋ?;8�clp?uWy��r?���T}?>	� �?���J�\�?-Ա�L�?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�������:�              �?              �?      �?      @      @      @       @      �?      �?      �?      �?              �?               @              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?              @      �?       @      �?        bj0xl&      ?LX�	��zo�A.*�L

lossR�^@

accuracyo�=
�
wc1*�	   ��뜿   ��̛?      r@!  ���F�?)H�Ȗ	�?2��v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�a�$��{E��T���C����#@�d�\D�X=���ڋ��vV�R9�>h�'�?x?�x�?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�u�w74?��%�V6?d�\D�X=?���#@?�!�A?�T���C?nK���LQ?�lDZrS?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?              �?      �?      �?       @      �?      @      @      @      @      @      @       @      @      @       @      @      @              @      @      �?      @      @      @      @      �?       @       @       @      @      �?       @      �?      @      �?      �?               @              �?              �?              �?              �?       @              �?              �?              �?              �?              �?              �?               @               @       @       @              @      @      @      @      @      @       @      @      @      $@      *@       @      $@      $@      @      "@      "@       @      ,@      &@      @      @      @      �?        
�"
wd1*�"	   �@y��    ���?      A!V��蠍@)��%v�M@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���
�}�����4[_>�����ӤP�����z!�?��T�L<���i����v��H5�8�t�Fixі�W���x��U���z!�?�>��ӤP��>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?�������:�              �?      @      @      ?@     �@@     �R@      `@     �@     ڠ@     Z�@     $�@     ��@     L�@    ���@     ��@    ���@    ���@    ���@     N�@    ���@    ���@     �@     0�@     q�@     ��@     ��@     F�@     D�@     �@     
�@     \�@     �@     r�@     ƨ@     @�@     ,�@     r�@     �@     �@     H�@     ��@     Ė@     <�@     Ԕ@     ԑ@     t�@     x�@     X�@     @�@     ȅ@     ��@     x�@     ��@     �~@     p{@      z@     �v@      v@     pt@     Pp@     �n@      m@     �k@     `i@     @d@      a@     �`@     @^@      X@      [@      Y@     �T@      S@     �N@     �P@     �L@     �G@      G@      F@      G@     �B@      @@      7@      1@      8@      $@      4@      0@      ;@      2@      $@      ,@      &@      &@      @      $@       @      @      (@      "@       @       @      @      @      @       @      @      �?       @      �?      @      �?      @               @       @      �?      �?       @      �?      @      �?              �?       @              �?              �?      �?              �?              �?               @              �?      �?               @      �?      �?      �?       @              �?               @      �?       @      @      @      �?      @       @      @       @      @      �?       @       @       @      @      "@      @      "@      "@      @      (@      &@      &@      (@      &@      ,@      &@      3@      ,@      4@      0@      2@      >@      <@     �A@     �A@      C@     �@@      E@     �L@      D@      N@     @P@     �P@     @T@     @R@      V@      ^@      a@     ``@     �`@     �c@     �f@     `f@     �j@     �k@      p@     r@     �s@     �t@     �u@     �x@     �y@     p@     �~@      �@     p�@     x�@     ��@     ��@     H�@     ��@     ,�@     ��@     �@     ��@     D�@     ��@     �@     2�@     8�@     ��@     T�@     ��@     �@     ��@     ذ@     ��@     
�@     c�@     ��@     �@     <�@     :�@    ���@    ���@    ���@     ��@     T�@     ��@    �9�@     ��@    ��@     �@     ��@     �@    ���@     �@     �@     �@     ��@     ��@     ��@      �@      k@     @P@      6@       @        
�
wo*�	   @*���   �ߥ?     ��@! 8�Tִ�)�й��?2�	�/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
��������|�~�>���]���>K+�E���>jqs&\��>�iD*L��>E��a�W�>�f����>��(���>�ߊ4F��>})�l a�>>�?�s��>�FF�G ?6�]��?����?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�������:�	               @      @      ,@      (@      9@      A@      J@      E@     �Q@     @R@      R@     �R@     @S@     �Q@      M@     �O@      K@      O@      D@      K@      J@     �K@      H@      A@      E@      ?@     �@@      6@      ;@      9@      7@      0@      ,@      ,@      $@      (@      $@      ,@      "@      $@      $@      @       @      @      @      @       @      @      @      @      @      @      �?      �?      @      �?              �?      �?      �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?      �?       @      �?      @      �?       @      �?      �?               @      @      @      �?      �?      @      @       @      @      @       @      @      @      (@       @       @      &@      @       @      ,@      ,@      ,@      0@      ,@      3@      >@      7@      4@      ?@      :@      A@      C@      H@      ?@     �H@      N@     �N@     �J@      I@     �H@      F@      K@      N@      M@     �D@      C@      F@      =@      :@      3@      @      @      �?               @        
�
bc1*�	   `(z�    9Ȟ?      @@!  ����?)�8־��?2ho��5sz�*QH�x�&b՞
�u�#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?�v��ab�?�/��?�������:h              �?      �?              @       @      �?      @       @       @              �?        
�
bd1*�	    fSx�   �Z��?      P@!  ����?)�_A3_��?2�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp����T}?>	� �?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              7@      @       @       @      @              �?              �?              �?      �?              @      @      @      @      @        
�
bo*�	    M��    '�?     �E@!  ��vWҿ)���t<�?2��/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���*QH�x�&b՞
�u���%>��:�uܬ�@8�uWy��r?hyO�s?���T}?>	� �?����=��?���J�\�?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�������:�              �?              �?      �?      @      @      @       @      �?      �?      �?      �?      �?              �?      �?      �?              �?              �?              �?              �?              �?              �?              �?      �?      �?               @      �?      @      �?        �)l�%      FN!�	uȥ{o�A/*�K

lossV�^@

accuracyڬz=
�
wc1*�	   �t)��   ��؛?      r@!  �H+v�?)XsG	8�?2��v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�a�$��{E��T���C��!�A���bȬ�0���VlQ.�ji6�9���.��ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?d�\D�X=?���#@?�T���C?a�$��{E?
����G?nK���LQ?�lDZrS?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?              �?      �?      �?       @      �?      @      "@      @      @      @      @       @      @      @      @      @      @              @      @      �?      @      @       @      @      �?       @       @       @      �?       @      @              @              �?              �?      �?              �?              �?              �?      �?      �?       @              �?              �?      �?              �?              �?       @              �?      @       @              @      @      @      @      @       @      @      @      @      (@      (@      "@       @      (@      @      "@       @       @      .@      $@      @      @      @      �?        
�!
wd1*�!	   �d�   @�'�?      A!�6
jP8�@){u�u\N@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��T�L<��u��6
��K���7��Łt�=	>��f��p>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>�4[_>��>
�}���>X$�z�>.��fc��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?�������:�              �?      @      @      B@     �F@     �T@      a@     ؅@     ��@     ��@     .�@     ˹@     j�@     ��@     ��@    ���@     ��@    ���@     �@     ��@     ��@     �@     J�@     V�@     j�@     ��@     o�@     C�@     ˲@     ۱@     {�@     �@     ~�@     x�@     ��@     .�@     Ȣ@     ,�@     �@     К@     ��@     �@     (�@     ��@      �@     `�@     (�@     @�@     @�@     ��@     ��@     �@     0�@     `~@     P|@     �y@     �v@      u@     �r@     �p@     @o@     �l@     �k@     �g@      d@     �b@      `@      _@     @Y@     @Y@     �W@      U@     �V@      M@     @P@     �N@     �H@     �E@      C@      E@     �C@     �A@      >@      ?@      ,@      0@      9@      1@      *@      1@      (@      ,@      3@      $@      (@      $@      @      @      @      *@       @       @      @      @       @      @      @      @      @      @       @              @              �?       @              @      �?      �?              @               @      �?              �?      �?              �?              �?              �?      �?              �?               @              �?              �?              �?      @      �?      @      �?       @      �?      @               @       @      @      @      @      �?              �?      @       @       @      @      $@      @      &@      (@      2@      $@      $@      ,@      0@      2@      3@      ,@      .@      8@      6@      ?@     �A@     �D@     �@@     �B@     �I@     �N@      O@      J@     �T@     �S@      S@     @W@      ^@      [@     @a@      a@     �c@     @f@     �g@     �g@     �j@      p@     �q@     �s@     `u@     `v@     0y@     py@     P~@     ��@     ��@     x�@     І@     Ȋ@     Ћ@     ��@     �@     ��@     �@     ��@     `�@     ș@     ��@     ��@     �@     �@     ,�@     ��@     $�@     *�@     \�@     ��@     8�@     O�@     �@     �@     	�@     ��@     �@     n�@    ���@     ��@    �`�@    �-�@     ��@    �/�@    ���@     �@    ���@     ��@     *�@    ���@     ��@     ��@     >�@     ʬ@     t�@     ��@     X�@      p@     @T@      =@      @        
�
wo*�	   �+���    �i�?     ��@! ]v�9��)�Q�]�C�?2�	�/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���vV�R9��T7����5�i}1���d�r�x?�x��豪}0ڰ>��n����>;�"�q�>['�?��>��(���>a�Ϭ(�>��Zr[v�>O�ʗ��>f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?�������:�	               @      @      ,@      *@      7@     �@@      K@      E@      S@     �Q@     @R@      T@     @Q@     �R@     �L@      O@     �M@     �M@      D@      K@     �J@     �J@     �G@      A@      F@     �A@      =@      4@      <@      8@      8@      1@      .@      (@      "@      0@      $@      .@      @      "@       @      @      @      @      @      �?      @      @      @      @       @      �?      @      @      @      �?               @               @              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?      �?              �?              �?               @      @      �?      �?       @      �?              @      �?      @              @      @       @       @      @      $@      �?      @      @      &@       @      $@      @      @      @      "@      0@      2@      .@      *@      6@      <@      4@      2@      @@      :@     �B@     �C@      F@      B@      H@      K@     �O@     �K@     �I@     �F@      G@     �J@     �M@     �L@     �C@      E@     �F@      ;@      8@      6@      @      @      �?              �?      �?        
�
bc1*�	   `/z�   �|	�?      @@!  �����?)t����?2ho��5sz�*QH�x�&b՞
�u�#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?�v��ab�?�/��?�������:h              �?      �?               @       @      �?      @      "@      "@              �?        
�
bd1*�	   ��Yx�   `�H�?      P@!  ��u�?)��w�
�?2�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp����T}?>	� �?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              7@      @       @      @       @              �?              �?              �?      �?              @      @      @       @      @        
�
bo*�	    }~��   `I��?     �E@!  ���ҿ)T�!"��?2��/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\��&b՞
�u�hyO�s�IcD���L��qU���I�hyO�s?&b՞
�u?o��5sz?���T}?����=��?���J�\�?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�������:�              �?               @      �?      @      @      @       @      �?      �?      �?      �?      �?      �?               @              �?              �?              �?              �?              �?              �?              �?      �?      �?               @      �?      @      �?        W��\%      Y���	@l[|o�A0*�J

lossf�^@

accuracyI��=
�
wc1*�	   ��_��   `2�?      r@!  �^M��?)�l�ÜM�?2��v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO��qU���I�
����G�a�$��{E��!�A����#@�U�4@@�$��[^:��"��u`P+d�>0�6�/n�>U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��%>��:?d�\D�X=?a�$��{E?
����G?nK���LQ?�lDZrS?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?              �?      �?      �?       @      �?      @      "@      @      @      @      @       @      @      @      @      @      @      �?      @      @      �?      @      @       @      @      �?       @      �?      @      �?       @      �?       @      @              �?      �?              �?              �?              �?               @              �?               @              �?              �?               @              �?       @              �?       @      @              @      @      @      @      @       @      @      @      @      &@      &@      "@       @      (@       @      "@       @       @      ,@      $@      @      @      @      �?        
�!
wd1*�!	   @wp��    ���?      A!��vF��@)8(�(�N@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g����u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K��R%������39W$:���X$�z��
�}�����4[_>���T�L<��u��6
��ہkVl�p�w`f���n�cR�k�e������0c������0c>cR�k�e>��z!�?�>��ӤP��>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?�������:�              @      @      .@     �@@      H@     �X@     @b@      �@     H�@     ̩@     M�@     �@     x�@    ���@    ���@     ��@     ��@    ���@     �@     ��@     ��@     ��@     F�@     =�@     a�@     ͸@     ��@     �@     ��@     ��@     @�@     �@     ��@     ��@     D�@     `�@     ʢ@     "�@     ��@     �@     К@      �@     ��@     ̔@     L�@     D�@     @�@     ��@     ��@     ��@     p�@     H�@     H�@     P|@     �{@     �y@     @v@     pt@     �r@     0p@     @o@     �j@     �k@     �g@     �d@      a@     �`@     �`@      ]@     @X@     �W@     �V@     �S@     �N@     �K@      G@     �I@     �C@      =@      F@     �E@     �B@      8@      6@      ;@      3@      5@      0@      2@      1@      0@      "@      ,@      .@      $@      "@      "@      @      &@       @      "@      "@       @       @      @      �?      @      �?      @      @              @       @      @       @      �?       @      �?      �?      �?              �?              �?       @              �?               @              �?              �?               @              �?               @       @              �?      @      �?      @      �?      �?       @      �?      �?      @       @      @      �?       @      @      @      @       @      @      @      @      "@      &@      @      *@       @      0@      0@      (@      "@      1@      &@      6@      3@      9@      8@      =@      ?@     �B@     �D@     �A@      G@      O@     �G@     @P@     �K@     @R@      Q@     �W@     �X@     �[@      ]@     �`@     �a@     `c@     @e@     �h@      h@     �i@     �q@     Pr@     Pt@     t@     Pv@     �w@     0z@     �~@     �@     0�@     �@     ��@     ��@     ��@     ؍@     ��@     �@     ȓ@     8�@     t�@     ؙ@     ��@     L�@     ��@     *�@     �@     ��@     (�@     ��@     �@     ��@     ��@     '�@     �@     ޷@     �@     Ի@     Ǿ@    �l�@     ��@    ���@     @�@     7�@     ��@    �%�@     V�@    ���@    ���@    ���@    �E�@     ��@     ?�@     W�@     M�@     �@     b�@     �@     �@     s@      \@     �C@      @        
�
wo*�	   ��|��   @��?     ��@! 'oom3�)s�-�?2�	�/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���vV�R9��T7����5�i}1���d�r��*��ڽ�G&�$��39W$:��>R%�����>��(���>a�Ϭ(�>})�l a�>pz�w�7�>6�]��?����?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?�/�*>�?�g���w�?�������:�	               @      @      ,@      *@      8@     �@@      L@     �E@      S@     �Q@     @R@     @T@     �Q@     �Q@      L@     �P@      L@      L@     �F@     �K@      I@     �K@      H@      >@      E@     �A@      @@      3@      ?@      8@      6@      .@      *@      *@      (@      3@      @      ,@       @      "@      $@      @       @      @      @       @      �?      @      @      @       @      �?      @      @      @              @              �?      �?      �?              �?      �?       @              �?              �?              �?              �?              �?               @      �?              �?       @       @      @      �?              �?      @       @      �?      @      �?       @      @       @       @      @      @      @      @      @      "@      @      $@      @      $@      @      &@      .@      1@      .@      2@      3@      :@      4@      8@      =@      7@     �C@      C@     �C@      F@     �E@      J@     �P@     �I@      K@      E@     �H@     �J@     �N@      I@      E@      F@     �E@      <@      7@      7@      @      @       @               @        
�
bc1*�	    R5z�   `e��?      @@!  @��G�?)��x���?2ho��5sz�*QH�x�&b՞
�u�#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?�v��ab�?�/��?�������:h              �?      �?              �?       @       @      @      (@      "@              �?        
�
bd1*�	    z_x�   �Ϟ?      P@!  ��D��?)�r{�Kt�?2�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp����T}?>	� �?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�������:�              7@      @       @      @       @              �?              �?              �?      �?              @      @      @      @      @       @        
�
bo*�	    v楿   ���?     �E@!  �ɒ�ҿ)j������?2��/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\��hyO�s�uWy��r�ܗ�SsW�<DKc��T�&b՞
�u?*QH�x?o��5sz?���T}?����=��?���J�\�?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�������:�              �?      �?      �?       @      @      @      @      �?      �?      �?      �?       @              �?      �?      �?              �?              �?              �?              �?              �?              �?              �?               @               @      �?       @       @        O1�'�%      G8K�	i}o�A1*�K

loss�g^@

accuracyI��=
�
wc1*�	   �����    R�?      r@!  �v��?){#��9[�?2��v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E����#@�d�\D�X=���VlQ.��7Kaa+��[^:��"?U�4@@�$?+A�F�&?I�I�)�(?��VlQ.?��bȬ�0?��82?�u�w74?uܬ�@8?��%>��:?
����G?�qU���I?nK���LQ?�lDZrS?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?              �?      �?      �?       @      �?      @      "@      @      @      @      @       @      @      @      @      @      @      �?      @      @      �?      @      @      �?      @      �?       @      �?      @      �?      @              @      @      �?              �?      �?              �?              �?              �?              �?              �?      �?       @              �?              �?              �?              @      �?              �?      @       @              @      @      @      @      @       @      @      @      @      &@      $@      $@       @      *@      @      "@       @       @      ,@      $@      @      @      @      �?        
�!
wd1*�!	   �����   ��W�?      A!��@�@)�����1O@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K��39W$:���.��fc���X$�z��
�}����ڿ�ɓ�i�:�AC)8g�=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>f^��`{>�����~>[#=�؏�>K���7�>��z!�?�>��ӤP��>�
�%W�>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?�������:�              @      @      5@     �@@      Q@      [@      d@     ��@     f�@      �@     \�@     #�@     ��@     ��@    ���@     ��@    ���@    ���@    ��@    ���@     ��@     �@     �@     �@     ��@     и@     {�@     ��@     ǲ@     ��@     <�@     ��@     ��@     (�@     �@     ��@     ��@     �@     ��@     ��@     ��@     X�@     ��@     L�@     @�@     �@     ȍ@     h�@     �@     �@     p�@     �@     p@     �@     @|@     �x@     �t@     �u@     @s@     �p@     @k@      n@     �j@      h@      d@      c@     @b@     �\@     @Z@     @W@     �Z@     �X@     @S@      N@      K@      M@      G@     �G@      F@      D@     �I@     �C@      ;@      =@      >@      5@      4@      *@      1@      3@      1@      *@      3@      *@       @      $@      @       @      @      ,@      @      @       @      @      @      @      @               @      @      �?      �?      �?      @      �?      �?              �?      @              �?               @              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              @               @               @       @      �?      �?      �?      @      @      @              �?      @      @      @      @      @      @      @      @       @      "@      &@      @      @      *@      .@      &@      (@      ,@      0@      2@      2@      5@      5@      5@      @@      C@      C@     �B@      E@     �F@     �J@     �N@      R@     �P@     @R@     �Y@     �Y@      Z@     �^@     �`@     �a@      c@     �d@     `j@     `i@      l@     �q@     �q@     ps@     �t@     pv@      w@     �{@     �}@     `�@     P�@     h�@     �@     `�@     H�@     ؍@     |�@     p�@      �@     �@     �@      �@     �@     ؞@     �@     ��@     ��@     ħ@     ܨ@     �@     ��@     ˰@     �@     �@     ��@     ��@     3�@     a�@     ��@    �i�@    ���@     ��@     �@     -�@    �y�@    ���@    �T�@     ��@    ���@     ��@     X�@     �@     d�@     ��@     ��@     ��@     �@     И@     ��@     �u@     �`@      I@      .@        
�
wo*�	    �ߤ�   @R��?     ��@! �vI�e�)V���?2�	�/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&��[^:��"��S�F !�ji6�9����ڋ��vV�R9��T7����5�i}1�x?�x��>h�'����~]�[Ӿjqs&\�Ѿ�XQ�þ��~��¾a�Ϭ(�>8K�ߝ�>�h���`�>��[�?1��a˲?f�ʜ�7
?>h�'�?�T7��?�vV�R9?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?�/�*>�?�g���w�?�������:�	              @      @      ,@      *@      9@     �A@      J@      K@     @Q@     �Q@     �R@      T@     @Q@     �Q@      L@     �P@      L@      K@     �G@     �L@      I@      L@      H@      ;@     �E@      A@      @@      4@     �@@      6@      5@      *@      0@      1@      &@      .@      (@      *@      @      @      $@      @      �?      @      @      �?       @      @       @      @       @       @      @       @      @              �?      �?               @      �?              �?       @       @              �?              �?              �?              �?      �?              �?              �?              �?               @      �?       @      @       @      �?      @               @       @      @      �?      �?       @      @       @      @      @      @      @      @      $@      @      "@      @      $@       @      @      7@      ,@      *@      0@      5@      ;@      4@      4@      ;@      :@      D@      C@      C@      E@      G@     �I@     @P@      I@     �L@     �D@     �I@      I@      N@      H@     �E@     �F@      E@      ?@      8@      5@       @       @      @               @        
�
bc1*�	   �;z�   �$W�?      @@!  @�U��?)��?���?2`o��5sz�*QH�x�&b՞
�u��7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?�/��?�uS��a�?�������:`              �?      �?               @      @      @      $@      &@              �?        
�
bd1*�	   ��dx�   ��S�?      P@!   jf�?)3� ��ԑ?2�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp����T}?>	� �?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�������:�              7@      @       @      @       @              �?              �?              �?      �?              @      @      @      @       @      @        
�
bo*�	    �N��   ��=�?     �E@!   ��3ӿ)�Qt�J�?2��g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\��uWy��r�;8�clp�E��{��^��m9�H�[�&b՞
�u?*QH�x?o��5sz?����=��?���J�\�?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�������:�              �?              �?      �?       @      @      @      @              �?      �?      �?       @      �?              �?      �?              �?              �?              �?      �?              �?              �?              �?               @              �?       @       @       @        �¸	l%      E��	�]�}o�A2*�J

loss�f^@

accuracy���=
�
wc1*�	   ��̝�   �)�?      r@! �1����?)�
�N`�?2��v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO��qU���I�
����G�a�$��{E����#@�d�\D�X=���%�V6��u�w74�ji6�9�?�S�F !?U�4@@�$?+A�F�&?��bȬ�0?��82?�u�w74?��%�V6?���#@?�!�A?�qU���I?IcD���L?nK���LQ?�lDZrS?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?              �?      �?      �?       @      �?      @      "@      @      @       @      @       @      @      @      @      @      @      �?      @      @      �?      @      @              @       @      �?      @      �?      @      �?      �?       @      @              �?      �?              �?              �?              �?              �?              �?              @              �?              �?              �?               @      �?               @      @       @      �?      @      @      @      @      @       @      @      @      @      &@      $@      $@       @      (@       @      "@       @       @      ,@      $@      @      @      @      �?        
� 
wd1*� 	   �����   �'�?      A! gq�)�@)3Iz�Z�O@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ����?�ګ�;9��R���5�L���u��gr��R%������
�}�����4[_>���T�L<��u��6
��K���7��[#=�؏���i����v��H5�8�t�BvŐ�r�ہkVl�p�4�e|�Z#���-�z�!������0c>cR�k�e>X$�z�>.��fc��>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�              @      @      >@     �C@      S@     �]@     �f@     �@     ��@     8�@     ��@     [�@     ��@    ���@    ���@    ���@     �@     ��@     �@     ��@     ��@     ݿ@     ��@     P�@     ��@     ϸ@     �@     �@     β@     ͱ@     �@     <�@     ��@     h�@     �@     ¤@     ��@     �@     �@     ��@     �@     ��@     ��@     P�@     ��@     (�@     ��@      �@     @�@     ��@     (�@     ��@     �@     �}@     �{@     �w@     v@      v@     s@     @r@      n@     �o@     @h@     �g@      e@     `a@      c@     �`@     �X@     �W@      Z@     �T@     �T@     �I@     �O@     �J@      I@      G@     �A@      H@     �G@     �@@      ?@      @@      ?@      9@      1@      3@      5@      .@      1@      @      (@      &@      @      @      (@      @      (@       @      @      @      @      @       @      �?      @       @      @      @       @       @               @              �?      �?              �?              �?               @      �?      �?              �?              �?              �?              �?              �?              �?       @              �?      @      @       @      @      @      �?      @               @      �?      �?      @      @      @      @      @       @      @      @      @      $@      "@      $@      0@      @      @      2@      .@      ,@      6@      1@      6@      <@     �C@     �@@     �F@      :@     �C@      G@     �G@     �I@     @Q@     @T@     �S@     @U@     @X@     @Z@      `@     @a@     �_@      d@     @f@      j@     @g@     `l@     p@     Ps@     �s@     Ps@     0v@     x@     �z@     @~@     �@     �@     ��@      �@     @�@     h�@     P�@     ��@     ��@     ��@     �@     |�@     �@     ��@     �@     
�@     �@     Ĥ@     Ц@     `�@     ��@     ��@     �@     �@     ��@     ��@     Ƿ@     �@     ��@     ]�@    ��@     ��@     ��@    ���@     �@    �Y�@    ���@    �P�@     ��@     ��@     ��@     T�@     ��@     ��@     ��@     ��@     r�@     ��@     �@     H�@     pz@     `e@     �M@      4@      �?        
�
wo*�	    :���   @��?     ��@! �*�܉�)I�^^��?2�	�/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�U�4@@�$��[^:��"��S�F !���ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'����(��澢f����E��a�Wܾ�iD*L�پ��>M|Kվ��~]�[Ӿ���m!#�>�4[_>��>���%�>�uE����>a�Ϭ(�>8K�ߝ�>6�]��?����?�T7��?�vV�R9?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?�/�*>�?�g���w�?�������:�	              @      @      0@      .@      7@      B@      J@      K@     �Q@      R@     �P@     @U@     �Q@     �Q@     �L@     @P@     �K@      L@      K@     �I@     �G@      K@      K@      6@     �F@      A@      @@      6@      ;@      ?@      ,@      ,@      3@      1@      (@      $@      "@      (@      @       @      "@      @       @      @      @      @      @       @       @      @      �?      �?       @      @      @              �?               @       @              �?      �?       @      �?              �?              �?              �?              �?              �?              �?              �?              �?               @              �?      �?       @      @       @      �?      �?              @       @      @              �?      @      @       @      @      @      @      @      @      *@      @      "@      @      &@      @      &@      4@      .@      (@      ,@      9@      7@      8@      2@      ;@      =@     �C@     �A@      C@      F@      H@     �I@     �N@      I@     �L@      D@      I@      K@     �M@     �G@      F@      E@     �F@      >@      6@      4@      (@       @      @               @        
�
bc1*�	   `G@z�    Z��?      @@!  @W��?)G�F��ω?2ho��5sz�*QH�x�&b՞
�u��7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�/��?�uS��a�?�������:h              �?      �?               @       @      @      $@      &@      �?              �?        
�
bd1*�	   �uix�    {П?      P@!  �~��?)��j*�?2�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp�o��5sz?���T}?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�������:�              7@      @       @      @       @              �?              �?              �?      �?              @      @      @      @      @      @        
�
bo*�	   `����   @���?     �E@!   5rӿ)�� �?2��g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\��;8�clp��N�W�m����%��b��l�P�`�&b՞
�u?*QH�x?o��5sz?>	� �?����=��?�7c_XY�?�#�h/�?���&�?^�S���?�"�uԖ?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?�������:�              �?              �?      �?      @      @      @      @              �?      �?       @      �?      �?      �?              �?              �?              �?              �?      �?              �?              �?      �?               @              �?       @       @      �?      �?        R�g7�%      Z\H�	!�~o�A3*�K

loss�.^@

accuracyKY�=
�
wc1*�	   ����   @�ԛ?      r@! ��Ws�?)�1���\�?2��v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO��qU���I�
����G�a�$��{E����#@�d�\D�X=�f�ʜ�7
?>h�'�?�S�F !?�[^:��"?�7Kaa+?��VlQ.?��bȬ�0?�u�w74?��%�V6?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?              �?      �?      �?       @      @       @      &@      @      @      @      @      �?      @      @      @      @      @      �?       @       @      �?      @      @       @      @       @       @       @       @       @       @      �?       @      @              �?      �?               @              �?              �?              �?      �?               @              �?      �?              �?               @      �?      �?       @      �?      @       @       @      @      @      @      @      �?      @      @      @      &@      &@       @       @      (@       @      "@       @      "@      *@      "@      @      @      @      �?        
�!
wd1*�!	   �W6��   ����?      A!tJ��t�@)��\��O@2�I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���X$�z��
�}�����4[_>������m!#��K���7��[#=�؏��4�j�6Z�Fixі�W�BvŐ�r>�H5�8�t>��ӤP��>�
�%W�>���m!#�>�4[_>��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�              �?      @      (@      @@     �H@      T@     �b@     �h@     8�@     ڡ@     ��@     ˵@     ��@     ��@    ���@     �@    ���@     ��@     ��@     A�@     ��@     ��@     ۿ@     ��@     ��@     y�@     ٸ@     F�@     ��@     ��@     ��@     �@     l�@     ��@     @�@     6�@     ��@     J�@     N�@     ��@     �@     ��@     h�@     ̕@     ��@     ԑ@     x�@      �@     p�@     �@     ��@     P�@      �@     0~@     �|@      {@     px@     �u@     �u@     �q@     pr@     Pp@     �n@     �h@     �g@     `b@     �c@     �b@      ^@     �\@     �V@     �Z@     �U@      U@      L@     �J@     �L@     �I@     �D@     �C@      F@      H@     �B@      <@      ?@      6@      ;@      7@      1@      4@      *@      .@      ,@      2@      &@      @      &@      "@      "@      @      @              &@      @      �?      @      @      @       @      @      @       @      �?       @      �?       @              @      �?               @      �?              �?       @              �?              �?              �?              �?              �?              �?      �?      �?              �?      �?              �?      �?              �?               @      @      @       @      �?       @       @      @      @       @      @       @      �?      @      @       @      @      @      @      @       @       @      $@      $@      ,@       @      *@      5@      .@      3@      9@      A@      9@      ?@      F@     �@@      @@     �C@      H@      H@     �R@     �K@     @S@     @S@      Z@     �W@     @Y@     @^@     @`@      a@     �b@     @f@     �g@     @h@     �i@     pp@     �r@      t@     `s@     �t@     �x@     �z@     `|@     �@     ��@     �@     p�@     p�@     �@     ��@     ��@     `�@     ��@     �@     ��@      �@     Ԝ@     ��@     :�@     ��@     ��@     �@     *�@     4�@     |�@     Ȱ@     �@     ȳ@     �@     ��@     Ϲ@     ��@     U�@    ��@    ���@    �|�@    ���@     ��@    � �@     ��@    ���@     ��@     ��@     _�@    �c�@    ���@     }�@     ��@     Ӵ@     �@     ��@     �@      �@     @~@     `h@     �S@      8@      @        
�
wo*�	   ��7��   �J��?     ��@! ����)�����?2�	�/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1��uE���⾮��%ᾙѩ�-߾E��a�Wܾ��~���>�XQ��>��~]�[�>��>M|K�>���%�>�uE����>a�Ϭ(�>8K�ߝ�>})�l a�>pz�w�7�>�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�	              @      @      .@      ,@      9@     �A@     �I@     �K@     @Q@      R@     �Q@      T@     @S@     �P@     �M@      P@     �J@      M@     �M@     �F@      H@     �K@      J@      ;@      D@      A@      @@      5@      <@      >@      0@      1@      3@      0@      $@       @      "@      &@      @      "@      &@      @       @      @       @       @      @       @      �?      @       @      �?      �?      @      �?      �?      �?      �?      �?               @      �?      �?               @      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?      @      @              �?              �?      @               @      �?      �?      @      @      @      @      @      @      @      @      $@      @      &@       @       @      @       @      3@      ,@      ,@      2@      8@      5@      3@      9@      9@     �@@     �A@      @@     �D@      E@     �H@     �J@      M@     �I@      K@      F@      I@     �K@     �L@      G@     �F@      C@     �H@      =@      8@      1@      ,@      �?      @      �?              �?      �?        
�
bc1*�	   `Ez�    ✢?      @@!   0X3�?)��m���?2ho��5sz�*QH�x�&b՞
�u��7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�uS��a�?`��a�8�?�������:h              �?      �?               @       @      @      $@      @      @              �?        
�
bd1*�	   ��mx�   �+$�?      P@!  @a�	�?)0�i.�r�?2�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp�o��5sz?���T}?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�������:�              7@      @       @      @       @              �?              �?              �?      �?               @      @      @      @      @      @        
�
bo*�	   ����    L��?     �E@!   n*�ӿ)j���̞?2��g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m�����J�\������=���ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed�hyO�s?&b՞
�u?o��5sz?���T}?>	� �?����=��?�7c_XY�?�#�h/�?���&�?^�S���?�"�uԖ?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?�������:�              �?              �?      �?      @      @      @      @              �?      �?       @       @              �?              �?              �?              �?              �?              �?              �?              �?      �?               @              @      �?       @      �?        ����,&      >@4?	YP�o�A4*�L

loss0+^@

accuracysh�=
�
wc1*�	   @�[��   �zÛ?      r@!  �۳+�?)���|�T�?2��v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��!�A����#@�6�]���1��a˲��5�i}1?�T7��?�vV�R9?��ڋ?�7Kaa+?��VlQ.?��82?�u�w74?��%�V6?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?              �?      �?      �?      �?      @      @      "@      @      @      @      @      �?      @      @       @      @      @      @      @       @      �?      @       @      @      @       @      �?       @      @              �?       @      @      �?              �?               @              �?              �?              �?              �?              �?              �?      �?              �?              �?      �?              �?      �?      �?       @      �?      �?      @       @      @      @      @      @      @      �?      @      @       @      "@      &@      "@      @      (@      @      $@       @      "@      *@      "@      @      @      @      �?        
�#
wd1*�#	   ��ݫ�   �W�?      A!@�ن��@)CyK�[P@2�I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#��[#=�؏�������~��i����v��H5�8�t���u}��\>d�V�_>�����0c>cR�k�e>BvŐ�r>�H5�8�t>E'�/��x>f^��`{>�����~>[#=�؏�>��z!�?�>��ӤP��>���m!#�>�4[_>��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�              @      @      5@      A@     �O@     @Z@     �a@     @m@     ��@     4�@     �@     ��@     ׺@     T�@    ���@     �@    ���@    ���@    ��@     ;�@    ���@     ��@     ƿ@     �@     �@     ��@     ˸@     >�@     ��@     ��@     "�@     @�@     �@     >�@     Ȩ@     ��@     z�@     �@     
�@     �@     ��@     ��@     (�@     ��@     h�@     �@     ȏ@     `�@     p�@      �@     ��@     �@     ��@     H�@      @     �{@      x@     �v@     `v@     @q@     �p@     p@     �l@     @k@      g@      d@     `d@     @a@     �\@     �[@      Y@     �X@     �T@     �T@      I@     @R@      K@      H@     �G@     �E@     �D@      D@     �B@      ?@      ;@      7@      0@      6@      3@      3@      &@      2@      .@      $@      0@      "@      "@      "@      @      $@      &@      �?       @      @      @      @      @      @      �?       @      @      �?      �?      �?       @               @      �?       @      �?              �?      �?      �?              �?              �?      �?      �?              �?              �?              �?              �?               @              �?              �?              �?              �?              �?      �?              �?              �?               @      �?      �?       @      �?      �?       @              �?       @      @              @      @      @      @      @      "@      @      @       @      @      "@      2@      (@      "@      $@      ,@      1@      6@      1@      1@      7@      :@      :@      F@      G@      >@      A@      H@      G@     @Q@     �L@      W@     �P@     @W@     �X@     �X@      _@      _@     �`@      d@     �e@     �i@      f@      j@     0p@     `r@      r@     0t@     �t@     Px@     �z@     �{@     ��@     ��@     ��@     ��@     p�@     Њ@     Ќ@     <�@     ȑ@     ��@     ,�@     D�@     �@     D�@     (�@     �@     ��@     ��@     z�@     ��@     N�@     ��@     ��@     �@     ��@     ��@     ��@     ��@     ��@     V�@     �@    �v�@     B�@     ��@     ��@    ���@    ���@    ���@     ��@    �k�@    �@�@    ��@     ��@     e�@     ��@     Ҵ@     �@     �@     �@     �@     P�@     @l@      X@     �C@      @        
�
wo*�	   �榿   �>�?     ��@! ���\��)���3?�?2�	�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&��[^:��"��S�F !���ڋ��vV�R9��T7�����(��澢f�����uE����0�6�/n�>5�"�g��>8K�ߝ�>�h���`�>�ߊ4F��>f�ʜ�7
?>h�'�?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�g���w�?���g��?�������:�	              �?      �?      @      ,@      0@      7@     �A@      K@     �J@     @Q@      R@      R@     �R@     �T@      P@     �M@      N@     �L@     �M@     �L@     �G@     �H@      K@      H@      ?@     �B@      A@      B@      5@      :@      ;@      3@      5@      1@      .@       @       @      @      (@       @      @      (@      @      @      @      @       @      @      @      @      @       @      �?      @      @              �?      �?      �?              @               @      �?              �?      �?              �?              �?      �?              �?              �?       @               @               @              @      @      �?      �?      �?      �?               @      @              @      @       @      @      @      @      @      "@      "@      @      &@      @      $@       @      $@      .@      0@      0@      3@      6@      5@      3@      ;@      8@      @@      B@     �@@     �D@      D@     �H@     �H@      O@      I@     �L@      F@      H@     �K@      L@      G@      H@      A@     �I@      ;@      ;@      .@      .@       @       @       @               @        
�
bc1*�	   �TIz�    �;�?      @@!   ��n�?)�[A�&;�?2ho��5sz�*QH�x�&b՞
�u��7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�uS��a�?`��a�8�?�������:h              �?      �?               @       @      �?      &@      @      @              �?        
�
bd1*�	   ��qx�   �]�?      P@!   �(�?)�!v[���?2�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp�*QH�x?o��5sz?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�������:�              7@      @       @      @       @              �?              �?              �?      �?               @      @      @      @      @      @        
�
bo*�	   �*���   �8Z�?     �E@!  ���ӿ)���V���?2��g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ����J�\������=���ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed�uWy��r?hyO�s?���T}?>	� �?�7c_XY�?�#�h/�?���&�?^�S���?�"�uԖ?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?�������:�              �?              �?      �?      @      @      @      @              �?       @      �?      �?       @              �?              �?              �?              �?               @              �?      �?               @               @      �?      @      �?        Y�l&      >:�	�_�o�A5*�K

loss�^@

accuracy�=
�
wc1*�	   @����   ൮�?      r@!  �k`��?)����K�?2��/����v��ab��}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A��[^:��"��S�F !�>h�'��f�ʜ�7
���(���>a�Ϭ(�>�[^:��"?U�4@@�$?��bȬ�0?��82?�u�w74?
����G?�qU���I?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?              �?      �?      �?      �?      @      @       @      @      @      @      @      �?      @      @      �?      @      @      @      @       @      �?      @      �?      @      @       @      �?      @       @      �?       @      �?       @      �?               @              �?              �?              �?              �?              �?              �?              �?      �?               @              �?              �?      �?      �?       @      �?               @      @      �?      @      @      @      @       @      �?      @      @      @      $@      &@       @       @      (@      @      &@      @      $@      (@      $@      @      @      @      �?        
�!
wd1*�!	   `@���   `곯?      A!�TS{6�@).:�CP@2�I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ������;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���
�}�����4[_>������m!#��K���7��[#=�؏��ڿ�ɓ�i�:�AC)8g�4�j�6Z>��u}��\>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>39W$:��>R%�����>�u��gr�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�              @       @      9@     �B@     @R@      `@     �e@     �o@     ��@     ��@     f�@     K�@      �@     m�@     �@    �J�@    ���@    ��@     ,�@     ;�@    �"�@    ���@     ˿@     F�@     m�@     ɺ@     ��@     1�@     ��@     �@     �@     =�@     ��@     |�@     ��@     ܥ@     �@     �@     ��@     �@     p�@     (�@     ��@     4�@     ��@     d�@     ��@     ��@     p�@     ��@     ��@     ��@      �@     ��@     �~@     0{@      y@     `w@     @t@     �p@     `p@     �o@     @m@     �g@     �f@      d@     �c@      a@      ^@      [@     @Z@     @[@     �S@     �P@      O@      N@     �E@      I@     �I@     �G@     �I@      @@      C@      <@      6@      :@      3@      ,@      5@      0@      ,@      .@      1@      3@      @       @      &@      "@      @      &@      $@      @      @      @      @      @      @      @              @      �?       @               @       @      @              �?               @      �?      �?              �?      �?              �?      �?              �?              �?              �?              �?              �?      @              @       @              @      �?       @      �?      @      �?      �?       @      @      �?      @       @      @      @      �?       @      @      @      @      @      @      "@      &@      "@      &@      $@      "@      4@      2@      .@      1@      &@      6@      8@      6@      ;@      A@      B@      B@      ?@     �B@     �J@     �D@     �O@     �L@     @T@     @Q@     �U@     @Y@     �\@     �[@      _@     ``@      f@     �f@     �g@      g@      m@     q@     �p@     �r@     �r@     �s@     Px@     |@      {@     ��@     ��@     8�@     Ȇ@     X�@     Ȋ@     �@     �@     đ@     |�@     �@     ��@     ��@     ��@     0�@     Ơ@     ¢@     �@     �@     ��@     8�@     2�@     ��@     #�@     ��@     ��@     ��@     ��@     ��@     �@    �!�@    �L�@     Q�@    ���@    ���@    ���@    ���@    ���@    �{�@     7�@     	�@     ��@     ��@     �@     ��@     ��@     -�@     P�@     �@     d�@     ��@     �p@      _@      G@      &@        
�
wo*�	   `I���   @(Щ?     ��@! O��p��)Ԇr2�d�?2�	�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0��7Kaa+�I�I�)�(��[^:��"��S�F !�ji6�9����ڋ��vV�R9��5�i}1���d�r�x?�x��>h�'���FF�G �>�?�s���O�ʗ����ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f���侞[�=�k���*��ڽ����%�>�uE����>8K�ߝ�>�h���`�>6�]��?����?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�g���w�?���g��?�������:�	              �?      �?      @      ,@      1@      8@      A@     �K@      K@     @P@     @R@     @R@      R@     �U@      O@      L@     �P@      J@     �Q@      H@     �G@      J@      J@      H@      >@      A@     �B@      >@      :@      ;@      <@      5@      1@      .@      1@      @       @      "@      (@      @      @      (@      @      @      @      @      @      @      �?       @      @      @      �?      @      @              �?              �?      �?              @              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @      �?      @      @      �?      �?      �?       @      �?      �?      @      �?      @      @      @      @      @      @      @       @       @      @      "@      @      &@      @      @      1@      1@      1@      0@      8@      5@      5@      7@      9@      =@     �D@      ?@      G@     �@@      H@     �L@      N@     �H@      J@      G@     �I@     �K@      L@     �G@     �F@     �@@     �J@      :@      ;@      0@      *@      @       @       @               @        
�
bc1*�	   �>Mz�    �ڣ?      @@!  �F#��?)̚�q=؋?2ho��5sz�*QH�x�&b՞
�u��7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�uS��a�?`��a�8�?�������:h              �?      �?              �?       @      �?      $@       @      @              �?        
�
bd1*�	   �Eux�    鑠?      P@!  @��A�?)�����?2�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp�&b՞
�u?*QH�x?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�������:�              7@      @       @      @       @              �?              �?              �?      �?               @      @      @      @      @      @        
�
bo*�	   �k駿    9��?     �E@!   ��
Կ)�����?2��g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ����J�\������=����N�W�m�ߤ�(g%k�5Ucv0ed����%��b�;8�clp?uWy��r?���T}?>	� �?����=��?�7c_XY�?�#�h/�?���&�?^�S���?�"�uԖ?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?�������:�              �?               @              @      @      @      @              �?      �?      @               @              �?              �?              �?              �?              �?      �?              �?      �?               @               @      �?       @       @        ��Q��%      Z\H�	���o�A6*�K

loss��]@

accuracytF�=
�
wc1*�	    ���   �ǚ�?      r@!  h��o�?)��$�G�?2��/����v��ab��}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L�
����G�a�$��{E��T���C��!�A���bȬ�0���VlQ.��7Kaa+�I�I�)�(�f�ʜ�7
�������5�i}1?�T7��?��bȬ�0?��82?�T���C?a�$��{E?
����G?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�������:�              �?              �?      �?      �?      �?      @      @      @      @      @      @      @       @      @      @      �?      @      @      @       @       @       @      @      �?      @       @       @       @      @              @               @       @      �?      �?              �?              �?              �?              �?              �?              �?               @              �?      �?               @              @      �?              �?      �?       @      @      �?      @      @      @      @       @       @      @      @      "@       @      &@       @      $@      "@      @      &@      @      $@      (@      $@      @      @      @        
�!
wd1*�!	   �^>��   @�6�?      A!�8�ߨ�@)>���VmP@2�I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L���u��gr��R%������.��fc���X$�z��
�}�����4[_>�����ӤP�����z!�?��T�L<��K���7��[#=�؏��H��'ϱS��
L�v�Q��`�}6D>��Ő�;F>d�V�_>w&���qa>�i����v>E'�/��x>f^��`{>�����~>K���7�>u��6
�>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�              @      (@      =@      N@      T@     �^@     �i@     @s@     �@     �@     �@     ��@     q�@     ��@     =�@    �q�@    ���@    �:�@    �)�@    �s�@    ��@     ��@     �@     [�@     ��@     ��@     ø@     �@     д@     #�@     ��@     N�@     ��@     N�@     H�@     ��@     Ĥ@     n�@     �@     ��@     ��@     ��@     4�@     ��@     �@     �@     x�@     �@     @�@     ��@     ��@     @�@     ��@     �@     �@     P{@     �w@     pu@     0v@     pq@     pp@     Pp@     `k@     �i@     �e@     �b@     �b@      `@     �\@      \@      \@     @Y@     �U@     �P@     �Q@     �M@     �K@      K@      L@      H@      D@      B@      ;@      ;@      >@      9@      7@      1@      3@      4@      $@      ,@      (@      *@      .@      @       @      @      @      "@      @      @      @      @       @      @      @      @              @       @      �?      @      @       @      �?      @      �?       @              �?              �?               @              �?      �?              �?              �?              �?              �?              �?      �?      �?              �?               @      �?              �?               @       @       @      �?              @      @       @      @      @      @      @      @      @      @       @      @      $@      &@      @       @      (@      *@      $@      *@       @      1@      <@      $@      0@      *@      9@      <@      ?@      C@      =@      <@      ;@     �H@      K@      N@     �Q@     �R@     �R@     �U@     �[@     @]@     �\@     �`@      `@     `e@     �f@     �g@     `g@     @j@     @p@     �q@     0s@     �t@     0t@     �w@     �z@      }@     X�@     ��@     h�@     p�@     ��@     Њ@     �@     �@     ��@     �@     ��@     ��@     ��@     ț@     d�@     ؠ@     �@     ��@     "�@     .�@     r�@     ��@     ��@     ��@     ��@     ��@     Y�@     ��@     f�@     �@     ޿@    �5�@    �C�@    ���@    ���@     p�@     ��@    �m�@    �;�@     �@    ���@    ���@     `�@     ׼@     ��@     ��@     9�@     ��@     �@     4�@     �@     @t@     �b@     �O@      1@        
�
wo*�	   �RJ��    �^�?     ��@! L�[��)OT�����?2�	�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��5�i}1���d�r�x?�x��>h�'��1��a˲���[���h���`�8K�ߝ�a�Ϭ(���(��澢f������>M|Kվ��~]�[Ӿ8K�ߝ�>�h���`�>})�l a�>pz�w�7�>�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�	              �?      �?      @      .@      0@      7@     �B@     �J@      K@     @P@      R@     �S@     �P@     �V@      P@      I@     �Q@     �H@     �R@      H@     �G@     �J@      J@     �E@     �@@      @@     �A@      @@      9@      =@      >@      0@      3@      *@      0@      &@      $@      @      (@      @      @      (@      @      @      @       @       @      @       @      @       @      @              @      @              �?      �?      �?               @       @              �?       @              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?       @       @      �?      @      @      �?              �?      �?               @      @      @       @      �?      @      "@      @      @      @      @      @      (@      @      @      @       @      ,@      2@      0@      .@      9@      8@      8@      3@      7@      @@     �B@     �B@     �C@     �B@     �G@      L@      O@      F@      K@     �G@     �I@     �L@     �L@      H@      E@     �A@     �G@      =@      =@      0@      *@      @      �?       @      �?               @        
�
bc1*�	    �Pz�   @�z�?      @@!   �s��?)�2�
n�?2`o��5sz�*QH�x�&b՞
�u��#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?`��a�8�?�/�*>�?�������:`              �?      �?              @      �?      @      &@      @              �?        
�
bd1*�	    �xx�    Ġ?      P@!  �z�W�?)�1?��?2�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp�&b՞
�u?*QH�x?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?�������:�              7@      @       @      @       @              �?              �?              �?      �?               @      @      @       @      @      @      �?        
�
bo*�	    �M��   ���?     �E@!   ��5Կ)-~�z�?2��g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ����J�\������=���;8�clp��N�W�m�E��{��^��m9�H�[��N�W�m?;8�clp?o��5sz?���T}?>	� �?����=��?�7c_XY�?�#�h/�?���&�?^�S���?�"�uԖ?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?�������:�              �?               @      �?      @      @      @      @              �?      �?      @               @              �?              �?              �?              �?              �?              �?              �?      �?               @               @      �?       @       @        �Tr�%      FN!�	Y�Ӂo�A7*�K

loss��]@

accuracy��=
�
wc1*�	   ��h��    ���?      r@!  ��F�?)��0]�L�?2��/����v��ab��}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L�
����G�a�$��{E��T���C�uܬ�@8���%�V6��u�w74��vV�R9��T7�������ž�XQ�þ��bȬ�0?��82?��%>��:?d�\D�X=?a�$��{E?
����G?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�������:�              �?              �?       @              �?      @      @      @      @      @      @      @       @      @       @      �?      "@       @      @       @       @      @      @       @       @       @      �?      �?      @      �?       @       @      �?      �?      �?      �?              �?      �?              �?      �?              �?              �?               @              �?              �?               @      �?      �?      �?      �?      �?      �?              @      @              @      @      @      @       @      @      @      @      "@      "@      $@       @      $@      "@      @      &@      @      "@      (@      $@      @      @      @        
�!
wd1*�!	   `�   �ʗ�?      A!�(ψ �@)���P@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ�������MZ��K���u��gr��R%������39W$:���X$�z��
�}�������m!#���
�%W����ӤP�����z!�?��[#=�؏�������~��
L�v�Q�28���FP�cR�k�e>:�AC)8g>K���7�>u��6
�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�������:�              �?      @      5@     �A@     �N@     �Y@      c@      k@     �t@     H�@     V�@     b�@     �@     ��@     ��@     \�@    ���@    ��@     V�@     >�@    ���@    ��@     ��@     6�@     1�@     ̼@     ��@     ��@     E�@     �@     ��@     ��@     s�@     ��@     ��@     Ĩ@     Υ@     Ƥ@     ��@     ��@     ��@     H�@     ��@     �@     h�@      �@     8�@     ��@     �@     0�@     ��@     Ѕ@     ��@     ��@     0@     �|@     �{@     �v@     �v@     �t@     �q@     Pp@      q@     �l@     �i@     @e@      c@     �c@     @_@     @]@     �Y@     �W@     @[@     @T@     @S@     @Q@     �O@     �G@     @P@     �K@      D@      C@      D@      =@      >@     �B@      1@      4@      4@      6@      0@      .@       @      .@      &@      $@      $@      @      "@      $@      @      @      $@      @      @      @      @      @       @       @       @      @      �?      @      @      �?      �?               @              �?               @              �?      �?      �?              �?              �?              �?              �?              �?      �?              �?      �?              �?      �?              �?              @       @       @      �?      @      �?      @      @       @      @      @       @      @      @       @      ,@      @      1@      @      "@      @      (@      &@      .@      &@      :@      1@      ,@      2@      3@      >@      >@      :@      D@      D@      @@      B@      K@     �H@      M@     �O@     @V@      R@     �W@      Y@     @[@     �X@     �a@     @a@     �d@     �e@     �h@      h@      i@     �p@     @r@     s@     pt@      t@     0z@     �{@      }@     ��@     8�@     X�@     h�@     (�@     @�@     p�@     d�@      �@     ̓@     ��@     (�@     �@     P�@     (�@      �@     ��@     t�@     ��@     ��@      �@     r�@     ��@     ��@     ��@     ~�@     η@     G�@     L�@     ҽ@     ��@    �=�@    �"�@    ���@     ��@     M�@     q�@     "�@    �B�@     ��@     ^�@     ��@     +�@     ��@     _�@     ��@     S�@     �@     N�@     �@     8�@     �w@     �e@      V@      4@      @        
�
wo*�	   @� ��    ��?     ��@! �����)�p�RR��?2�	���g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9������6�]���1��a˲���[���ߊ4F��h���`�8K�ߝ뾮��%ᾙѩ�-߾�h���`�>�ߊ4F��>�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�	              �?               @      @      0@      .@      8@     �B@     �J@      L@     �O@      R@     �R@      R@     @U@      Q@      J@     �P@     �J@     �Q@      G@      G@     �L@     �J@      E@      <@     �B@      =@      A@      :@      ?@      ;@      ,@      5@      0@      3@       @      @       @      *@      @      @      $@      @      @      @      �?      @      @       @      @      @      @              @      @              �?              �?      �?               @      �?               @      �?              �?              �?              �?      �?              �?              �?              �?      �?       @      �?               @      @      @      �?              @      �?              @      @      @      @      �?      @      @      @      @      $@       @      @      $@      @      @      @      @      0@      2@      (@      3@      6@      ;@      7@      4@      :@      <@     �C@      C@     �A@      E@     �F@      J@     �O@      G@      K@      G@      K@      K@      L@     �I@     �D@      B@      E@      @@      :@      4@      &@      @       @       @      �?               @        
�
bc1*�	    Tz�   @^�?      @@!  @��?)"Շ��?2`o��5sz�*QH�x�&b՞
�u��#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?`��a�8�?�/�*>�?�������:`              �?      �?              @      �?      @      $@       @              �?        
�
bd1*�	   �t{x�   `���?      P@!  ���l�?)5� iG�?2�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp�hyO�s?&b՞
�u?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?�������:�              7@      @       @      @       @              �?              �?      �?              �?               @      @      @       @      @      @      �?        
�
bo*�	   @ܰ��    �?     �E@!   ��^Կ)b�C�ՠ?2����g�骿�g���w��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�����=���>	� ��uWy��r�;8�clp�ܗ�SsW�<DKc��T�ߤ�(g%k?�N�W�m?*QH�x?o��5sz?����=��?���J�\�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?^�S���?�"�uԖ?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�              �?               @      �?      @      @      @      @              �?       @       @      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?               @               @      �?       @      �?      �?        ���%      FN!�	%ʌ�o�A8*�K

loss��]@

accuracy㥛=
�
wc1*�	   �⾟�   ����?      r@! ��sS��?)l�:��_�?2��/����v��ab��}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L�
����G�a�$��{E��T���C��!�A�d�\D�X=���%>��:�uܬ�@8��vV�R9��T7����5�i}1���d�r���bȬ�0?��82?�u�w74?��%�V6?a�$��{E?
����G?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�������:�              �?              �?       @              �?      @      @      @      @      @      @       @       @      @      �?      �?      "@      @      @      @       @      @      @      @      @      @      �?      �?      @      �?       @      �?      �?               @      �?              �?              �?              �?      �?              �?              �?              �?      �?      �?              �?              �?       @              �?              �?      �?       @              �?       @      @       @      @      @       @      @      @       @      @      @       @      "@      &@      @      "@      $@      "@      "@      @      "@      (@      $@      @      @      @        
�!
wd1*�!	    ����   `���?      A!��	m��@)�����P@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ����]������|�~���MZ��K���u��gr��.��fc���X$�z���4[_>������m!#��w`f���n>ہkVl�p>E'�/��x>f^��`{>�����~>[#=�؏�>
�}���>X$�z�>39W$:��>R%�����>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�������:�              @       @      6@      H@     @Q@      ^@     �d@     �n@     �v@     ��@     ��@     ¬@     @�@     �@     �@    �n�@     ��@    �&�@    �\�@     m�@    ���@    ��@    ���@    �!�@     o�@     ��@     Ӻ@     ۸@     a�@     ʴ@     �@     	�@     ��@     \�@     Ԫ@     T�@     ��@     ��@     ��@     �@     ��@     �@     D�@     �@     0�@     \�@     |�@     ��@     `�@     Њ@     �@     x�@     p�@     ȁ@     �@      ~@     p{@      x@     pw@     �t@     @q@     @q@     pp@     �h@      i@     `f@      c@     �b@     @b@     @^@     �Z@      Y@     @W@      U@     �U@     �K@     �N@     �G@      L@      C@      E@      F@      B@      C@      A@      <@      8@      2@      .@      0@      .@      &@      0@      ,@      (@       @      ,@      $@      @      @      @      $@      @      @      @      @      @      @      @      @      @      @               @      �?       @      �?       @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?      @       @       @      @      @       @       @      @              @      �?       @      @      @      @      @      @      *@      @       @      "@      "@      &@       @      &@      ,@      "@      (@      ,@      2@      1@      *@      =@      >@      8@      @@     �C@      E@      8@     �C@     �H@      H@     �P@      R@     @T@     �T@     @T@      [@      Z@     �Y@     `a@     �`@     @e@      h@     �f@     �g@     �j@     0r@     �r@     ps@      t@     �u@     x@     z@     �}@     �@     Ѐ@      �@     ��@     (�@     ��@     @�@     $�@     t�@     X�@     �@     ��@     ��@     t�@     �@     �@     �@     |�@     <�@     ��@     n�@     r�@     y�@     n�@     Ƴ@     ��@     y�@     2�@     �@     ��@     }�@    �0�@     �@    ���@     J�@     I�@     �@     �@    ���@     ��@     ;�@     ��@    ��@     M�@     O�@     ��@     S�@     ^�@     ��@     X�@     ؇@     �|@     �i@     @Z@      ?@      @        
�
wo*�	   @񶩿   �em�?     ��@! �򇄠�)IS�����?2�	���g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ�})�l a��ߊ4F��h���`���(��澢f���侮��%ᾙѩ�-߾�h���`�>�ߊ4F��>>h�'�?x?�x�?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?���g��?I���?�������:�	              �?              @      @      1@      *@      7@     �D@      H@     �N@      N@      R@     �R@      R@     �T@      Q@     �K@     �P@      L@     �P@     �F@     �I@      M@      I@      D@      ;@      D@      :@     �A@      :@      ?@      :@      3@      .@      1@      1@      &@       @       @      (@      @      @      &@      @       @       @      �?      @      @      �?      @      @       @      �?      @      @      �?       @      �?      �?              �?      �?       @              @              �?      �?              �?              �?              �?              �?              �?       @              �?              @       @      @      �?               @      �?      �?               @      @      @      @       @      @      @      @      @      "@      @      @      (@      @      $@      @       @      *@      0@      0@      2@      9@      9@      3@      9@      6@      =@     �D@     �A@      C@     �B@     �H@     �G@     �P@      H@     �I@     �I@     �K@      J@      K@     �I@      C@      D@     �D@      @@      ;@      2@      &@      @      @       @      �?               @        
�
bc1*�	    �Vz�   @�ʥ?      @@!   �B7�?)��|�כֿ?2`o��5sz�*QH�x�&b՞
�u��#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?`��a�8�?�/�*>�?�������:`              �?      �?              @      �?      @       @      $@              �?        
�
bd1*�	    !~x�   �z(�?      P@!  @\z��?)C���(z�?2�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp�uWy��r?hyO�s?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?�������:�              7@      @       @      @       @              �?              �?      �?              �?               @      @      @      @      @      �?      @        
�
bo*�	   ���   �"�?     �E@!   eh�Կ)J$ K�1�?2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�����=���>	� ��hyO�s�uWy��r�k�1^�sO�IcD���L�P}���h?ߤ�(g%k?*QH�x?o��5sz?���J�\�?-Ա�L�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?^�S���?�"�uԖ?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�              �?              �?      �?       @      @      @      @      @               @      �?       @      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?               @              �?       @       @      �?      �?        VՀ�\&      c^u	�?�o�A9*�L

loss�X]@

accuracy䃞=
�
wc1*�	   ����   @Ǌ�?      r@! �@%L��?)�6���?2��/����v��ab��}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��!�A����#@���%>��:�uܬ�@8��S�F !�ji6�9��f�ʜ�7
�������.�?ji6�9�?�u�w74?��%�V6?uܬ�@8?��%>��:?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�������:�              �?              �?       @      �?       @      @       @       @      @      @      @       @       @      @      �?       @      "@      @      @       @      @      @      @      @      @       @      �?      �?      @       @               @              �?      �?      �?      �?              �?               @              �?              �?              �?              �?              �?              �?              �?       @              �?              �?              �?       @      �?               @       @       @       @      @       @       @       @      @      �?      @      @      @      $@      $@       @      @      *@       @      $@      @      "@      (@      $@      @      @      @        
�"
wd1*�"	   `$_��   �8_�?      A!<ie�@)��L�Q@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L����|�~���MZ��K���u��gr��X$�z��
�}����u��6
��K���7��[#=�؏�������~�BvŐ�r�ہkVl�p�w&���qa�d�V�_�w`f���n>ہkVl�p>�����~>[#=�؏�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�������:�              @      (@      :@     �O@     �S@     �`@      g@     Pq@      y@     |�@     0�@     ��@     N�@     /�@     O�@     ��@    ���@    �%�@    ���@     n�@     ��@     ,�@     ��@     %�@     q�@     ׼@     ��@     ˸@     ��@     ��@     �@     �@     >�@     �@     �@     b�@     ֥@     ̤@     v�@     �@     �@     �@     ��@     ��@     ��@     ��@      �@      �@      �@     (�@     ��@     ��@     p�@     ��@     @     `|@     �{@      w@     x@     �t@     �r@     @q@     Pp@     �k@     `j@     �e@     @d@     �d@     �`@     @`@     @[@      W@      X@     �T@     �U@      P@      I@      I@      M@     �I@      B@     �H@     �C@      C@      <@      =@      8@      ;@      2@      *@      .@      2@      1@      2@      $@      ,@      @       @      @      $@      $@      "@      @      @      @      �?      @      @      @      @      @       @       @      @      @       @       @       @      �?      �?               @       @               @              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?       @              �?      �?              �?       @              �?      @              �?       @      @      @       @      @       @       @      @      @      @       @      &@      "@      @      @      @       @      $@      1@      .@       @      *@      2@      0@      ;@      4@      6@      9@      =@      E@      <@      >@     �B@      K@      O@      J@      Q@     @U@     �R@     @U@      Y@     �[@     �\@     �`@     �a@      c@      e@     �h@     �g@      k@     �p@     �r@     �s@     �t@     `r@      y@     �y@     �}@     ؀@     ��@     ��@     ��@     �@     (�@     X�@     t�@      �@     8�@     p�@     ��@     ؘ@     4�@     d�@     Π@     ��@     Z�@     ��@     ��@     ګ@     0�@     u�@     ͱ@     Գ@     �@     U�@     X�@     �@     r�@     l�@    ���@      �@     [�@     F�@     ��@    ���@    ��@     ��@    ���@     �@     S�@     ��@     �@     o�@     ��@     ��@     ��@     0�@     ��@     ��@     �~@     `o@      a@      C@      (@        
�
wo*�	   ��o��    o�?     ��@!  .q���))*d��*�?2�	���g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(��[^:��"��S�F !�ji6�9���.����ڋ���d�r�x?�x��pz�w�7��})�l a��ߊ4F��8K�ߝ�a�Ϭ(龋h���`�>�ߊ4F��>>�?�s��>�FF�G ?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?���g��?I���?�������:�	              �?      �?       @      @      1@      ,@      7@      C@     �I@      M@      O@     �P@      T@     @R@     �T@     �P@     �M@     �P@      J@     �P@      H@      I@      M@     �G@      D@      <@      D@      ;@     �@@      <@      >@      <@      .@      ,@      2@      1@      (@      "@      $@      "@      @      @      (@      @      �?      @      �?      @      @      @      @      @      @      �?      @      @              �?      �?              �?               @      @              �?              �?      �?              �?               @              �?              �?      �?       @      �?              @      �?       @      @      �?       @      �?              �?      @       @      @      @      @      @      @      @      @      $@      @      @      (@      @       @      @      "@      ,@      1@      *@      2@      <@      7@      6@      2@      9@      >@     �E@      @@      C@      C@     �F@     �I@     �N@      H@      K@     �G@      M@     �J@      K@      J@     �C@     �B@      F@      8@     �B@      ,@      &@       @      @       @      �?               @        
�
bc1*�	    �Yz�    7|�?      @@!  �f�o�?)���|gf�?2`o��5sz�*QH�x�&b՞
�u��#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�/�*>�?�g���w�?�������:`              �?      �?              @      �?      @       @      $@              �?        
�
bd1*�	   `��x�    �\�?      P@!  @����?)1�E4��?2�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp�uWy��r?hyO�s?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?�������:�              7@      @       @      @       @              �?              �?      �?              �?              @      @      @      @      @      �?      @        
�
bo*�	    Ht��    �B�?     �E@!  �>��Կ)�}Mh���?2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�����=���>	� ��hyO�s�uWy��r��!�A����#@�5Ucv0ed?Tw��Nof?&b՞
�u?*QH�x?���J�\�?-Ա�L�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?^�S���?�"�uԖ?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�              �?              �?      �?       @      @      @      @      @               @      �?       @      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?               @              �?       @      �?       @      �?        �0�\&      c^u	ʤ��o�A:*�L

loss"f]@

accuracy	��=
�
wc1*�	   �*��   �	��?      r@! �Ch���?)K��Α��?2��/����v��ab��}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6�U�4@@�$��[^:��"�I��P=��pz�w�7��6�]��?����?��%>��:?d�\D�X=?���#@?�!�A?�qU���I?IcD���L?k�1^�sO?nK���LQ?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�������:�              �?               @      �?      �?       @      @      @       @      @      @      @       @       @      @      �?       @      "@      @       @      @      @      @      @      @      @      @               @      �?       @      �?      �?      �?      �?      �?              �?      �?      �?              �?              �?              �?              �?              �?              �?              @              �?              �?      �?      �?              �?       @      �?               @       @      �?       @      @      @       @      @      @      �?      @      @      @      "@      $@      @      $@      &@      $@      $@      @      $@      (@      "@      @       @      @        
�"
wd1*�"	   @q��   `�ű?      A!��jm�F�@)>G��[Q@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ����?�ګ�;9��R���5�L�����]������|�~���u��gr��R%������39W$:���.��fc���X$�z�����m!#���
�%W��u��6
��K���7��[#=�؏����u}��\�4�j�6Z�:�AC)8g>ڿ�ɓ�i>[#=�؏�>K���7�>�
�%W�>���m!#�>X$�z�>.��fc��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�������:�              @      6@      @@      L@     �Z@     @c@     �i@     �s@     Pz@     X�@     ��@     H�@     t�@     u�@     h�@    ���@    ���@     N�@    ���@     m�@    ���@     *�@    �#�@     �@     Ǿ@     ��@     ��@     �@     ��@     ƴ@     ��@     �@     �@     j�@     Ȫ@     ��@     ��@     l�@     \�@      �@     �@     ��@     ę@     ��@     T�@     �@     ̑@     ��@     X�@     �@     x�@     Є@     ��@     ��@     P@     ~@     �z@     �w@     �x@     �t@      r@     0q@     @n@     �k@     �i@      d@     �d@      d@     �a@      `@     �Z@     �V@     �X@     �V@     @S@      O@     �J@     �O@     �L@      E@      A@     �I@     �B@     �C@      <@      :@      <@      1@      *@      6@      1@      1@      $@      ,@      (@      *@      &@      (@      @       @      $@      @      $@      @      @       @      @      @       @      @      @      @      �?      �?      �?      �?              �?       @      �?      �?              �?      �?               @              �?              �?      �?              �?              �?              �?              �?              �?              �?               @      �?      �?      �?      �?      �?      �?      �?              �?              @      �?       @      �?       @      @      @      @      @      @       @      @      @      @      $@      $@      &@      *@      *@      *@      *@      ,@      6@      ,@      1@     �@@      ;@     �@@      B@      E@     �A@     �A@     �L@     �M@      M@     @P@     �T@     �R@     �U@     �[@     �Y@     �[@     @_@     @a@     �d@     �f@     �f@      h@     �i@     �q@     �p@      s@     �r@     �s@     `y@     @y@     �{@     ��@     ��@     ��@     Ȇ@      �@     h�@     ��@     ��@     $�@     ��@     �@     d�@     ��@     ��@     ��@     ��@     ~�@     t�@     n�@     R�@     �@     �@     $�@     ��@     ��@     !�@     G�@     ϸ@     %�@     R�@     �@    ���@     �@     N�@    ��@    ���@    ���@      �@     ��@    �u�@     ��@    �
�@     ��@     ��@     b�@     ��@     ��@     ��@     ��@      �@     p�@     ��@     �r@     `c@      P@      1@        
�
wo*�	   �'��   ``��?     ��@! �Z��)"��v�?2�	I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1�pz�w�7��})�l a�h���`�8K�ߝ���>M|K�>�_�T�l�>�h���`�>�ߊ4F��>�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?�������:�	              �?              �?       @      @      1@      *@      ;@     �B@      H@      N@     @P@     �P@     �S@     �R@     �R@     �Q@      O@     �P@     �H@      Q@      I@      H@     �L@      G@     �D@      =@      D@      =@      ?@      6@     �A@      8@      2@      .@      1@      1@      $@       @      $@      "@      @      @      $@      $@      @      @       @       @      @       @      @      �?      @      �?      @      @               @              �?      �?      �?               @      @       @      �?      �?               @              �?              �?              �?              �?       @              �?               @      �?       @      @      �?      @              �?       @      @       @      @      @      @      @      @      @      "@      @       @      $@      @      "@      @      $@      &@      3@      .@      1@      ;@      8@      7@      2@      9@      >@     �D@      @@      D@     �A@      E@     �J@      O@     �I@      I@      G@     �M@      L@      K@      J@      C@     �A@     �F@      9@     �B@      ,@      $@      "@      �?      @              �?               @        
�
bc1*�	   `\z�   `�4�?      @@!  ���?)O;]o�5�?2ho��5sz�*QH�x�&b՞
�u��#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�/�*>�?�g���w�?�������:h              �?      �?               @       @      @      $@      $@      �?              �?        
�
bd1*�	   �x�   �Ĕ�?      P@!   (X��?)4k�9�?2�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp�;8�clp?uWy��r?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?�������:�              7@      @       @      @       @              �?              �?      �?              �?              @      @      @      @      @      @      @        
�
bo*�	    ԩ�    )��?     �E@!  01��Կ)=���F��?2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/��#�+(�ŉ�eiS�m��>	� �����T}�&b՞
�u�hyO�s�ji6�9���.�����%��b?5Ucv0ed?hyO�s?&b՞
�u?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?^�S���?�"�uԖ?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�              �?              �?      �?       @      @      @      @      @              @              @              �?              �?              �?              �?              �?              �?              �?              �?              �?               @              �?       @      �?       @      �?        ���L&      ?fjd	�Ϯ�o�A;*�L

loss0L]@

accuracy�=
�
wc1*�	   ��L��     ��?      r@!  �5w�?)n#��?2��/����v��ab��}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L�
����G�a�$��{E��T���C��!�A����#@���82���bȬ�0�+A�F�&�U�4@@�$��[^:��"��MZ��K�>��|�~�>+A�F�&?I�I�)�(?��VlQ.?��bȬ�0?�!�A?�T���C?a�$��{E?
����G?IcD���L?k�1^�sO?�lDZrS?<DKc��T?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?               @      �?      �?       @      @      @      @      @      @      @      @      @      @       @      �?       @      @      @       @      @       @      @      @      @      @              �?       @               @      �?               @      �?      �?              �?       @              �?               @              �?      �?              �?              �?              �?              �?              �?              @              �?               @      �?       @       @              @      @      �?      @      @      @       @      @      @       @      *@      $@      @       @      *@      &@       @      @      &@      &@      $@      @      "@      @      �?        
�!
wd1*�!	   ��h��    �-�?      A!O���J�@)s���Q@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���u��gr��R%������39W$:���.��fc���X$�z��
�}�����
�%W����ӤP�����z!�?��T�L<��u��6
����x��U�H��'ϱS�ڿ�ɓ�i>=�.^ol>�H5�8�t>�i����v>f^��`{>�����~>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?�������:�              �?      @      8@     �H@      S@     �W@     �f@     �k@     �u@     �|@     đ@     D�@     X�@     �@     ��@     |�@     ��@     ��@    ��@    ���@     `�@     ��@     =�@    ��@     6�@     ��@     ��@     ��@     �@     B�@     ۴@     ߲@     ڱ@     �@     j�@     ��@     6�@     ��@     ��@     ��@     ؠ@     l�@     @�@     ș@     �@     L�@     ��@     ȑ@     (�@     Ѝ@     ��@     ��@     (�@      �@     ��@     P~@     |@     Pz@     Px@     �w@     `u@     �q@     �p@     �o@     @n@     �j@      f@      c@     @b@     �b@     @Y@     �X@     @X@      V@     �T@      V@     �O@     �M@     @P@     �H@     �E@     �D@     �K@      A@      E@      4@      7@      3@      1@      4@      4@      5@      @      &@      2@      0@      "@      *@      "@      @      &@      @       @      @      @      @       @       @      @      @      �?      @      @      �?       @              �?      �?               @               @              �?              �?               @              �?      �?              �?              �?              �?              �?              �?      �?               @              �?      �?               @      �?              �?       @      @      �?      @       @      @       @      @      @      @      &@      @      $@       @      @      @      (@      0@      "@      $@      7@      0@      .@      *@      4@      ;@      >@      :@     �D@      C@     �@@      E@      N@      I@      K@     @P@     �S@     �S@      V@     �V@      [@     �Y@      a@      `@     �d@     �e@     @j@      g@     @j@     �q@      r@     s@     �s@      s@     �y@     pz@     |@     @�@     0�@     P�@     ��@     h�@      �@     @�@      �@     $�@     �@     ��@      �@     �@      �@     \�@     Р@     ��@     У@     Ʀ@     ��@     ��@     ޭ@      �@     �@     9�@     !�@     �@     ��@     �@     3�@     ��@    ���@     �@    �1�@    ���@     ��@     X�@     ��@    ��@    �D�@     ��@     ��@     ��@     ܻ@     U�@     0�@     �@     �@     ��@     ��@     ��@     ��@     @v@      g@     �W@      1@       @        
�
wo*�	   @߫�   �I�?     ��@! ��>��)�	���?2�	I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"�ji6�9���.����ڋ�I��P=��pz�w�7��})�l a��ߊ4F��h���`�>�ߊ4F��>I��P=�>��Zr[v�>>�?�s��>�FF�G ?��[�?��d�r?�5�i}1?�T7��?�vV�R9?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?�������:�	              �?              �?      @      @      1@      0@      7@      C@      H@      O@     @P@     @S@     �Q@     @R@     �Q@      S@      P@     @P@      I@     �P@      H@     �I@     �I@     �F@     �G@      <@      C@      >@      A@      4@      >@      ;@      0@      2@      .@      2@      &@       @       @      $@      @      @      $@      "@      @      @       @      @      @      @      @      �?      @       @      @      @              �?              �?       @      �?               @      @              �?      �?      �?              �?              �?              �?      �?              �?               @              �?              @      �?      �?      @               @              �?      �?      @      @       @      @      @      @      @      @      @      @      @      @      (@      @       @      @      @      *@      4@      *@      1@      :@      9@      :@      0@      9@      >@      D@      >@      E@     �@@      F@      J@      O@      G@      K@      G@     �K@      O@      K@     �I@     �B@      B@      F@      ;@      A@      *@      ,@      @      @      @      �?      �?               @        
�
bc1*�	   �>^z�    ��?      @@!  @���?)R�ٞ��?2ho��5sz�*QH�x�&b՞
�u��#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�/�*>�?�g���w�?�������:h              �?      �?              �?      @      @      "@      (@      �?              �?        
�
bd1*�	   �x�   ��͡?      P@!  �GU��?)'Oj�e5�?2�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp�;8�clp?uWy��r?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?�������:�              7@      @       @      @       @              �?              �?      �?              �?              @      @      @      @       @      @      @        
�
bo*�	   �>2��    S�?     �E@!  p$�տ)���(M�?2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/��#�+(�ŉ�eiS�m��>	� �����T}�*QH�x�&b՞
�u���82?�u�w74?E��{��^?�l�P�`?uWy��r?hyO�s?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?^�S���?�"�uԖ?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�              �?              �?      �?      @      �?      @      @      @              @              @              �?              �?              �?              �?              �?              �?              �?              �?      �?               @              �?      �?      �?      @      �?        l)p�&      =(�R	78f�o�A<*�M

loss�]@

accuracyO�=
�
wc1*�	    �r��   ���?      r@!  �Vg�?)\cN40^�?2��/����v��ab��}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��!�A����#@�d�\D�X=���%>��:�uܬ�@8��u�w74���82�I�I�)�(�+A�F�&��[^:��"��S�F !��5�i}1���d�r�O�ʗ��>>�?�s��>��%�V6?uܬ�@8?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?               @      �?      �?       @      @      @      @      @      @      @      @      @      @       @      �?      $@       @      @       @      @       @      @      @       @      @       @              �?      �?      �?      �?              �?       @      �?              �?      �?              �?       @              �?              �?              �?              �?              �?              �?              �?              �?              �?       @               @              @      �?       @              @      @      �?      @       @      @      @      @      @      @      $@       @      "@      @      .@      &@       @      @      $@      &@      (@      @      "@      @      �?        
�"
wd1*�"	   ��ǰ�   ����?      A!�KJ�0d�@)n��~R@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d��豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���u��gr��R%������39W$:���.��fc���X$�z��
�}������ӤP�����z!�?��K���7��[#=�؏�������~�f^��`{�ڿ�ɓ�i�:�AC)8g�Fixі�W���x��U�p
T~�;>����W_>>w`f���n>ہkVl�p>f^��`{>�����~>K���7�>u��6
�>�4[_>��>
�}���>X$�z�>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?�������:�              @      &@      9@      O@     �T@     ``@     �e@      o@     y@     @~@     @�@     ��@     ̮@     1�@     ��@     ��@     ��@    ��@    ���@    ���@     d�@     ��@     C�@     1�@    ��@     ��@     n�@     ��@     ׸@     ;�@     �@     ��@     �@     ��@     ��@     �@     ^�@     ֥@     ��@     n�@     x�@     �@     �@     @�@     @�@     �@     ԓ@     (�@     ��@     `�@     p�@     ��@      �@     (�@      �@     �~@     �|@     `{@     0x@     0w@     pu@     �r@     �q@     �n@     �k@     @i@     �f@     @f@      a@     @_@      _@     @W@     �U@     @X@     @Y@     �T@     �Q@     �L@     �Q@     �L@      H@     �B@     �E@      C@     �B@      9@      9@      8@      4@      :@      7@      &@      (@      &@      .@      1@      $@       @      "@      @      @      "@       @      @      @      @      @      @      @      @       @      @      �?      @      �?              �?      �?      �?       @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?              @      �?       @       @      @      @       @       @       @      @      @      "@       @      @      "@      @      "@      @      &@      "@      "@      1@      &@      "@      4@      0@      ,@      ,@      3@      :@      8@      ?@     �A@     �@@      ;@     �E@      I@      D@      J@      R@     �R@      Q@      V@     �V@     �W@      [@     �^@     @c@      e@      d@     �g@     �h@      k@     @o@     �q@     �r@      t@     0u@      y@     �y@     �{@     @�@     `�@     `�@     ��@     `�@     P�@      �@     ��@     ��@     ��@     X�@     ��@     4�@     |�@     ��@     ��@     ��@     Z�@     ��@     �@     X�@     ʭ@     ?�@     ڱ@     	�@     .�@     ��@     ʸ@     ��@     -�@     s�@    ���@     ��@     Q�@    ���@    ���@    �%�@    ���@     S�@     ��@     ��@     ��@    ���@     λ@     9�@     ֵ@     +�@     (�@     �@     h�@     ��@     h�@     �z@      m@     �\@      <@      @        
�
wo*�	   `����   @���?     ��@! p�w���)�qv"m+�?2�	I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ�>h�'��f�ʜ�7
�6�]���1��a˲�I��P=��pz�w�7��})�l a�h���`�>�ߊ4F��>x?�x�?��d�r?�T7��?�vV�R9?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?I���?����iH�?�������:�	              �?      �?              @       @      *@      2@      8@     �B@     �J@     �J@      R@     �S@      P@     �S@      Q@     �S@     �N@      P@     �L@      N@      H@     �I@     �I@     �H@     �F@      ;@     �B@      >@     �A@      1@      ?@      ;@      0@      *@      6@      1@      &@       @      "@      $@      @      "@      "@      @      @      @       @       @      @      �?      @      �?      @       @      @      @      �?      �?              �?      �?              �?      @       @              �?              �?               @      �?              �?              �?               @              �?      �?       @       @      �?      @              @              �?      @       @       @      @      @      @      @       @      @      $@      @      @      "@      @       @      @       @      .@      2@      1@      4@      3@      8@      ;@      5@      7@      =@      C@      :@     �D@      C@     �F@     �G@     �O@      H@      M@     �D@      K@     �O@     �K@     �H@     �D@      B@     �C@      >@      =@      3@      (@      @      @      @       @      �?               @        
�
bc1*�	   �<`z�    a��?      @@!   `.9�?)�_4:��?2po��5sz�*QH�x�&b՞
�u��#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�g���w�?���g��?�������:p              �?      �?              �?      @      @      "@      (@              �?              �?        
�
bd1*�	   ���x�    S
�?      P@!  ��@�?)�-���|�?2�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m?;8�clp?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?�������:�              7@      @       @      @       @              �?              �?      �?              �?              @       @      @      @      @      @      @        
�
bo*�	   �o���    �_�?     �E@!  ��[1տ)�+Pꭢ?2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/��#�+(�ŉ�eiS�m��>	� �����T}�*QH�x�&b՞
�u�a�$��{E?
����G?��bB�SY?�m9�H�[?;8�clp?uWy��r?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?^�S���?�"�uԖ?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�              �?              �?      �?      @      @       @      @      @      �?       @              @              �?              �?              �?              �?              �?              �?              �?      �?      �?               @               @      �?       @       @        ��o_&      >:�	0���o�A=*�K

lossf�\@

accuracyV}�=
�
wc1*�	   �>���   @f�?      r@!  �E��?)�˧�>ɤ?2��/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A�d�\D�X=���%>��:���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�ji6�9���.���ѩ�-�>���%�>�[^:��"?U�4@@�$?�!�A?�T���C?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?              �?      �?      �?       @       @      @      @      @      @      @      �?      @      @      @       @      �?      @      @              @      @      �?      @      @      @      @      @              �?      �?      �?               @      �?      �?               @      �?              �?               @              �?              �?      �?              �?              �?              �?              �?              �?       @      �?               @              �?      @       @              @      @      @      @      @      @      @      @      @      @      $@      @      $@      @      *@      &@      @      @      $@      $@      *@      @      @      @      �?        
�!
wd1*�!	   �)��   ���?      A!׾�lT��@)�nq�nR@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R�����]������|�~���MZ��K��R%������39W$:���.��fc���X$�z���4[_>������m!#���
�%W����ӤP�����z!�?���i����v��H5�8�t�R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?�������:�              @      ,@      B@     �O@     �W@     �c@     �f@     Pq@     @z@     ��@     $�@     8�@     ��@     ��@     #�@    ���@     ��@     3�@     }�@     ��@     n�@     ��@     8�@    �'�@     1�@     z�@     ��@     ��@     ��@     L�@     ��@     ��@     ��@     ��@     ��@     ��@     ��@     r�@     ��@     N�@     d�@     D�@     ��@     ��@     H�@     X�@     ̓@     ��@     8�@     h�@     ��@     ��@     ؅@     ��@     H�@     p}@     @~@     �{@     �w@     �v@     �t@     @r@     �q@     @o@      j@     �h@     �f@      d@     �c@     �a@      \@     @W@     @W@     @Y@     �W@     �T@     �P@      M@     �N@     �J@      G@      F@     �F@     �C@     �C@      4@      5@      0@      7@      7@      5@      3@      $@      $@      .@      *@      1@      "@      ,@      "@      @      &@      @      @      @      @      �?      @      @       @       @      @      @      �?      @       @      @              �?      �?              �?      �?              @              �?              �?      �?              �?              �?              �?      �?      �?       @              �?      �?      �?      @      �?      @      @      �?              @      �?      @      �?      @      @      @       @      "@       @      &@      @       @      ,@       @      ,@      $@      *@      &@      .@      0@      2@      4@     �@@      0@      :@      >@     �D@     �@@      J@      E@     �G@      G@     �R@     �Q@      R@      Q@      W@     @X@     �]@      a@     �`@     �d@      e@      f@     �g@     �j@     �o@     �q@     `r@     �s@     s@     Py@     �y@     �{@     h�@     ��@     H�@     ��@     8�@     �@     ��@     �@     ��@     <�@     ̔@     |�@     ��@     L�@     <�@     ��@     Z�@     P�@     ��@     ܧ@     8�@     ԭ@     G�@     ��@     �@     ��@     ��@     ��@     ��@     �@     4�@    ���@    ���@     R�@    �d�@    ���@    ���@    ���@    ��@    ��@    ���@    ���@    ���@     ��@     .�@     =�@     n�@     ��@     ֣@     P�@     <�@     Ї@     0�@     �p@     �a@     �C@      "@        
�
wo*�	   ��I��    ���?     ��@! �Kj4��)M�U��?2�	I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ���[���FF�G ���Zr[v��I��P=��pz�w�7���ߊ4F��h���`iD*L��>E��a�W�>��(���>a�Ϭ(�>�h���`�>�ߊ4F��>6�]��?����?�T7��?�vV�R9?��ڋ?�.�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�	              �?      �?      �?      @      "@      (@      2@      6@      E@     �I@      J@     �R@     �R@      P@     �T@     @P@     @T@      M@      Q@     �L@      L@      G@     �I@     �K@      H@      F@      ;@      C@      <@      B@      1@      >@      9@      3@      *@      3@      3@      ,@      $@      "@      &@      @      @      &@      @      @      @      �?      @      @      �?      @      �?       @      �?      @      @      �?      @       @              �?       @              �?      @      �?              �?              �?      @              �?              �?              �?              �?              �?               @              �?              @       @      �?      @       @       @       @               @       @      @      @      @      @      @      @      @      @      @      @      (@      @      @      @       @      2@      1@      3@      3@      4@      7@      6@      9@      6@      >@     �A@      =@      C@      D@     �E@     �H@      N@      I@     �J@     �E@     �K@     �L@      O@      G@     �E@      B@      C@      ?@      <@      3@      ,@      @      @       @      @              �?               @        
�
bc1*�	   �bz�    �u�?      @@!  ��@��?)��!���?2ho��5sz�*QH�x�&b՞
�u����&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�g���w�?���g��?�������:h              �?      �?              @       @       @      ,@              �?              �?        
�
bd1*�	   �9�x�   ��H�?      P@!  �#�&�?)I����Ŕ?2�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m?;8�clp?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?�������:�              7@      @       @      @       @              �?              �?      �?              �?              @       @      @      @      @      @      @        
�
bo*�	   �]調   �C��?     �E@!  ��]տ)ą%��?2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/��#�+(�ŉ�eiS�m�����T}�o��5sz�*QH�x�k�1^�sO?nK���LQ?<DKc��T?ܗ�SsW?�N�W�m?;8�clp?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?^�S���?�"�uԖ?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�              �?              �?      �?      @      @       @      @      @       @      �?      �?       @              �?              �?      �?              �?              �?              �?               @              �?               @               @      �?       @       @        Q�u�&      `���	��K�o�A>*�M

loss�\@

accuracy4��=
�
wc1*�	    �ՠ�   �I�?      r@!  ��@�?)G�S�D�?2��uS��a���/�����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T�nK���LQ�k�1^�sO�IcD���L�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���VlQ.��7Kaa+��f�����uE����+A�F�&?I�I�)�(?uܬ�@8?��%>��:?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?              �?       @               @       @      @      @       @      @      @      �?      @      @      @              @      @      @      �?      @      @       @      @      @      �?      @       @       @      �?       @               @              �?      �?              �?      �?       @              �?              �?      �?      �?              �?              �?               @              �?              �?       @      �?              �?              �?      �?              �?      �?      @      �?       @      @      @       @      @       @      @      @      @      @      @      &@       @      @      $@      *@      "@      @      "@      $@      &@      @       @      @       @        
�#
wd1*�#	    ~���   `n�?      A!�VH��@)'��q�R@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L���MZ��K���u��gr��R%������39W$:���.��fc���X$�z����ӤP�����z!�?��u��6
��K���7�������~�f^��`{�E'�/��x��i����v�:�AC)8g>ڿ�ɓ�i>w`f���n>ہkVl�p>[#=�؏�>K���7�>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?�������:�              @      7@     �G@     @T@     @Y@     �d@      j@     �s@     �z@     8�@     @�@     ��@     ��@     �@     ;�@     ��@     ��@     S�@    ���@     ��@    ���@    ���@    �B�@      �@    �D�@     *�@     ̼@     ��@     ��@     0�@     ��@     ��@     ڱ@     ��@     ��@     ��@     ��@     ��@     F�@     `�@     `�@     ��@     ��@     \�@     <�@     T�@     ��@     �@     X�@     ��@     ��@     ��@     ��@     `�@      �@      ~@     P}@     @z@     w@     @w@     0t@     �p@     �q@     �o@      k@     �f@      e@     @d@     �c@     �\@     @_@      Y@     �W@     �X@     �U@     @T@     @R@     �P@     �F@      N@      H@     �@@      H@      @@     �B@      8@      >@      :@      7@      9@      8@      $@      ,@      ,@      2@      0@      @      "@      @      @      "@      $@       @      @       @      @      @      �?      @      @       @      @      @      @      �?       @              @              �?      �?              �?       @              �?       @              �?              �?              �?               @               @              �?              �?              �?       @              �?      �?               @      �?       @               @              @      �?      �?      �?      @              �?               @      @       @      @       @      @      @      @      &@      @      @      &@      @       @      *@      $@      $@       @      $@      1@      4@      0@      7@      2@      =@      5@      7@      @@     �D@      A@      G@     �H@     �I@     �I@     �P@      S@     �R@     �U@     �R@     @Z@     �X@     @a@     @b@     �d@     �d@      g@      h@      j@      p@     `r@     `s@     ps@     0t@     Py@     �z@      {@     �@     ��@     8�@     ��@     ��@     ��@     Ȍ@     ��@     $�@     0�@     ��@     8�@     L�@     |�@     ��@     @�@     <�@     `�@     �@     "�@     F�@     ��@     <�@     `�@     �@     ��@     ��@     ;�@     }�@     ּ@     �@    �n�@     ��@    ���@     s�@     `�@     ��@    ���@     ��@     ��@    ���@    ���@    ���@     ��@     �@     J�@     �@     0�@     �@     �@     ̓@     ��@     ��@     �t@     �d@     �P@      .@        
�
wo*�	   �����   ��\�?     ��@!  ���)>�,lv��?2�	����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9���Zr[v��I��P=��pz�w�7���ߊ4F��>})�l a�>6�]��?����?f�ʜ�7
?>h�'�?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�	              �?              �?       @      @      &@      *@      1@      5@     �E@     �J@      H@     �U@     �O@     @P@     @T@     @Q@      U@     �K@     �P@      M@     �J@     �H@      I@      K@      I@      D@      =@     �C@      :@      @@      7@      =@      ;@      1@      1@      5@      ,@      ,@       @       @      $@      @      @      &@       @      @      @       @       @      @      �?      @       @      @       @      @       @      @      @              �?              �?              �?      @       @      �?               @      �?              �?              �?              �?               @               @       @              @       @      @      @      �?      �?      �?       @      �?       @       @      @       @       @       @      @      @      @      @       @      @      &@      @      &@       @      "@      3@      5@      1@      .@      4@      8@      6@      9@      5@      ;@     �B@      <@      D@      C@      D@      J@     �K@     �J@     �L@     �B@     �N@      L@      M@     �H@     �D@      A@      E@      =@      ?@      1@      ,@       @      @       @      @              �?               @        
�
bc1*�	    �cz�    b9�?      @@!   c���?)u���~�?2ho��5sz�*QH�x�&b՞
�u����&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�v��ab�?�/��?�g���w�?���g��?�������:h              �?      �?              @       @      "@      ,@              �?              �?        
�
bd1*�	   ฉx�   @N��?      P@!   ��I�?)�p�?2�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m?;8�clp?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�������:�              7@      @       @      @       @              �?              �?      �?              �?              @       @      @      @      @      @       @      �?        
�
bo*�	   ��?��    ��?     �E@!  �8�տ)fѝ��o�?2�I�������g�骿�/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/��#�+(�ŉ�eiS�m�����T}�o��5sz�*QH�x�IcD���L?k�1^�sO?ܗ�SsW?��bB�SY?ߤ�(g%k?�N�W�m?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?^�S���?�"�uԖ?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?�������:�              �?               @              @      @       @      @      @       @      �?      �?       @              �?              �?      �?              �?              �?              �?              �?      �?      �?               @               @      �?       @      �?      �?        ���H,'      8p�g	���o�A?*�N

loss�w\@

accuracy�T�=
�
wc1*�	    ���   @���?      r@!  J�5��?)rIGcϥ?2��uS��a���/�����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO��qU���I�
����G�a�$��{E��T���C�uܬ�@8���%�V6���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(��vV�R9��T7�����VlQ.?��bȬ�0?��%�V6?uܬ�@8?d�\D�X=?���#@?a�$��{E?
����G?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?              �?       @              @      �?      @      @      @      @      @      @      @      @      @              @      @       @      @      @      @      @       @      @      @      @      @       @       @      �?      �?               @              �?              �?      �?       @               @              �?      �?              �?              �?              �?              �?              �?              @              �?      �?              �?              �?              �?      �?              @      @      �?      @       @      @       @      @      @      @      @      @      @      (@      @      @      &@      (@      $@      @      $@      &@       @      $@       @      @       @        
�"
wd1*�"	    F�   �ڳ?      A!��8ݤ�@)����ES@2��{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]�����MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}������ӤP�����z!�?��T�L<��u��6
��f^��`{�E'�/��x��H5�8�t�BvŐ�r�cR�k�e������0c��
�%W�>���m!#�>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?�������:�              �?      @      ?@     �L@      W@      _@      f@     �k@     w@     p{@     ��@     d�@     �@     �@     �@     ��@    ���@     �@    �i�@     ��@     ��@    ���@     ��@    �7�@     ;�@    �,�@     2�@     �@     ��@     ϸ@     !�@     ��@     ��@     ��@     ¯@     8�@     Ъ@     ��@     ��@     6�@     p�@     ��@     X�@     d�@     h�@     ��@     ̔@     T�@      �@     �@     ��@     `�@     (�@     X�@     H�@     ��@     @~@     �|@     �{@     0w@     Pw@     �s@     `o@     �q@     �m@      j@      h@     �e@     �b@     �b@     �`@     �`@      Y@     @V@     @Z@     �W@     �R@      R@     �L@      J@      N@      F@      A@     �G@      E@      ;@      9@      8@      =@      .@      ;@      (@      0@      ,@      *@      .@      *@      &@      .@      @      @      $@      @      @      @      *@       @      @      @       @      @      @      @      @      �?      �?       @       @       @      �?       @      @       @              �?       @              �?              �?              �?              �?              �?              �?              �?              �?               @      �?      �?       @              �?              �?       @       @      �?      @       @              �?      @      �?      @              @      @      �?      @       @      @       @      "@      @      @      (@      "@      ,@      &@      @      5@      "@      7@      4@      6@      =@      4@      ;@      ?@      F@      F@     �E@      H@      I@     �L@     @Q@     �P@     @Q@      U@     @U@      [@     �Z@     �a@     @b@      d@     �e@     @f@     �g@     @l@     @o@     pq@     �s@     `r@     s@     �x@     �z@     P{@     `@     p�@     Ђ@     ��@     X�@     P�@     ��@     \�@     t�@     0�@     �@     P�@     �@     ��@     ��@     "�@     ��@     �@     ^�@     ʧ@     8�@     �@     �@     '�@     �@     ʹ@     @�@     ��@     m�@     e�@     ؽ@     i�@     ~�@     ��@     C�@    ��@    ���@    ���@    �u�@    ���@     Q�@     ��@     ��@     i�@     �@     F�@     \�@     ��@     ��@     ��@     ܕ@     h�@     @�@      w@      j@     �W@      3@      �?        
�
wo*�	   @����   `�	�?     ��@! �;�c*�)��l�?2�	����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r���Zr[v��I��P=���h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�������:�	              �?      �?               @      @      &@      .@      2@      4@      E@      L@     �G@      V@     �M@     �Q@     �T@     �O@     �U@     �J@      Q@      K@      L@     �I@      F@      L@     �H@     �D@      >@      C@      <@      ;@      9@      =@      =@      3@      4@      ,@      (@      *@      &@      $@      $@      @      @      $@      @      @      @       @       @      @       @      @       @      @      �?      @       @       @      �?      �?      �?      �?      �?              @       @      �?              �?      �?              @              �?      �?      �?      �?              �?      �?               @              �?              �?      �?      �?       @      @      �?      �?              �?       @       @      @       @      @      @      @      @      @      $@      @      @      (@      @      "@      @      (@      (@      3@      ,@      2@      5@      :@      6@      :@      2@      ;@      C@      ;@      E@     �B@     �B@      J@      L@     �J@     �M@     �A@      M@     �O@      K@      I@     �C@      B@     �E@      :@      @@      3@      &@       @      "@              @              �?               @        
�
bc1*�	   �-ez�   @���?      @@!  �9��?)���h��?2ho��5sz�*QH�x�&b՞
�u����&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�v��ab�?�/��?���g��?I���?�������:h              �?      �?               @      @      "@      ,@              �?              �?        
�
bd1*�	   ��x�   ����?      P@!   ]�h�?)ib�loO�?2�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp�ߤ�(g%k?�N�W�m?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�������:�              7@      @       @      @       @              �?              �?      �?              �?              @       @      @      @      @      @      @      �?        
�
bo*�	   @
���   ��k�?     �E@!   �տ)���(�ϣ?2�I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/��#�+(�ŉ�eiS�m�����T}�o��5sz�*QH�x��T���C?a�$��{E?�m9�H�[?E��{��^?P}���h?ߤ�(g%k?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?^�S���?�"�uԖ?}Y�4j�?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?�������:�              �?              �?      �?      �?      @      @       @      @       @       @      �?      �?       @              �?              �?      �?              �?              �?              �?              �?      �?      �?              �?      �?               @      �?       @      �?      �?        �w�b�'      :�BW	�5��o�A@*�O

loss�*\@

accuracy�Q�=
�
wc1*�	   �Kr��   �0��?      r@!  ���?)$��1h�?2��uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E�uܬ�@8���%�V6��u�w74�+A�F�&�U�4@@�$���ڋ��vV�R9�f�ʜ�7
�������vV�R9?��ڋ?+A�F�&?I�I�)�(?d�\D�X=?���#@?�!�A?�T���C?�qU���I?IcD���L?k�1^�sO?nK���LQ?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?              �?               @      �?       @      @      @      @      @      @      @      @      @      @      @       @      @      @       @       @      @      @       @      @      @      @      @              �?      @       @               @      �?              @              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?      �?       @               @              �?              �?      �?               @               @      @      @      @      @               @      @      @      @      @      @       @       @      $@      @      &@       @      ,@      $@       @      "@      &@      "@      "@       @      @       @        
�#
wd1*�#	    2]��   ��F�?      A!�5�C{�@)c�ȽŵS@2��{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP���T�L<��u��6
��K���7��/�p`B�p��Dp�@�6��>?�J>������M>ڿ�ɓ�i>=�.^ol>��ӤP��>�
�%W�>���m!#�>
�}���>X$�z�>.��fc��>39W$:��>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?�������:�              @      (@      A@     �P@     @X@      c@     @i@     �m@      y@     �}@     Ѕ@     ��@     N�@     ��@     ?�@     �@     ��@     @�@    �}�@    ���@    ���@    ���@     �@     P�@     .�@    �$�@     L�@     ��@     ��@      �@     ��@     x�@     ɲ@     ��@     ��@      �@     ��@     ا@     ��@     ��@     8�@     f�@     О@     X�@     H�@     ��@     T�@     ��@     ȑ@     0�@     ��@     �@     `�@     @�@      �@     h�@      }@     P~@     `z@     �w@     �v@     �t@     Pp@     pq@     �o@     @l@      g@      h@     `b@     �c@     �^@     @Z@      Z@     �T@     �Y@     �X@      S@     @Q@      I@     �I@      N@      D@     �G@      H@      ?@      @@      7@     �A@      4@      <@      3@      1@      .@      (@       @      *@      1@      $@       @      &@      @      @      @      @      @      @      "@      @      @      @      @       @      @      @      �?       @      �?               @               @               @      �?      �?              �?              �?      �?              �?              �?      �?              �?      �?              �?              �?              �?              �?      �?              �?      �?      �?              �?      @      �?              �?      �?      @       @      �?      �?       @       @       @      @      @      �?      @      @       @      "@      @      &@      $@      @      @      @      @      1@      &@      @      *@      0@      1@      ,@      1@      :@      A@      3@      8@      A@     �D@      F@     �@@      J@      M@      N@     �M@      U@     �U@     �U@     �S@     @Y@      Z@     �`@     @a@     `c@     �d@     �d@     �g@     �j@     �o@     pp@      t@     `r@     �s@     @y@     py@     �|@     �@     X�@     `�@     ؅@      �@     ��@     h�@     l�@      �@     ��@     Ȕ@     8�@     �@     ��@     0�@     8�@     H�@     ��@     �@     ܧ@     0�@     �@     ��@     8�@     �@     {�@     V�@     l�@     �@     J�@     ��@     H�@     B�@     ��@    �&�@     	�@    ���@     e�@    �9�@     o�@     :�@     _�@     ��@     '�@     �@     /�@     ʲ@     R�@     L�@     П@     ��@     8�@     8�@     P|@     �n@     �^@      7@      @        
�
wo*�	   ��b��   �2f�?     ��@! �k�4�)������?2�	����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��O�ʗ�����Zr[v��I��P=��G&�$�>�*��ڽ>a�Ϭ(�>8K�ߝ�>�ߊ4F��>})�l a�>x?�x�?��d�r?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�������:�	              �?      �?               @      @      *@      &@      4@      5@      F@     �J@     �G@     @V@     �N@      R@      S@      R@      T@      J@     �Q@     �J@     �M@     �H@      E@     �J@     �I@      D@      =@     �D@      ;@      =@      :@      ;@      ;@      6@      4@      (@      ,@       @      *@      $@      (@      @      @      &@       @       @      @      �?      @      @      �?      @      �?       @      @      @       @       @       @      �?              �?              �?       @       @              �?              �?      �?              �?       @              �?              �?              �?              �?               @              �?              �?      �?      �?      @      @      �?       @       @              �?      @      @      @      @      @       @      @      @      @      @      @      (@      &@      @      $@      @       @      &@      2@      ,@      3@      6@      9@      9@      6@      5@      8@     �B@      ?@      E@      A@     �C@     �G@     �N@     �J@      L@      C@      J@      P@      N@      H@      C@      @@     �G@      9@      A@      1@      &@      $@      @       @      @      �?              �?               @        
�
bc1*�	   ��fz�   ���?      @@!    �I�?)�_F!q�?2ho��5sz�*QH�x�&b՞
�u����&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�/��?�uS��a�?���g��?I���?�������:h              �?      �?               @      @      "@      ,@              �?              �?        
�
bd1*�	   �O�x�   `���?      P@!  ��M��?)�r/�y��?2�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp�P}���h?ߤ�(g%k?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�������:�              7@      @       @      @       @              �?              �?      �?              �?              @      �?      @      @      @      @      �?      @        
�
bo*�	   �m嫿   `4��?     �E@!  X���տ)IM���.�?2�I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/��#�+(�ŉ�eiS�m�����T}�o��5sz�*QH�x���bȬ�0?��82?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?�"�uԖ?}Y�4j�?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?�������:�              �?              �?      �?      �?      @      @       @      @      @               @               @              �?              �?      �?              �?              �?              �?              �?      �?      �?               @               @      �?      �?       @      �?        ���'�&      <���	��w�o�AA*�M

loss��[@

accuracy7�=
�
wc1*�	   ��ס�   �H�?      r@! �H���?)�P���?2��uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO��qU���I�
����G�a�$��{E�d�\D�X=���%>��:��vV�R9��T7����5�i}1���d�r��[^:��"?U�4@@�$?�7Kaa+?��VlQ.?��bȬ�0?��82?uܬ�@8?��%>��:?d�\D�X=?���#@?
����G?�qU���I?IcD���L?nK���LQ?�lDZrS?<DKc��T?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?              �?               @       @       @      @      @      @      @      @      �?      @      @      @      @      �?      @      @      @      �?      @      @       @       @      @      @       @       @              �?      @      @      �?      �?      @      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?               @              �?       @              �?      �?               @       @      �?              @      �?      @      @              �?      @      @      @      @      @      �?       @       @       @      &@      @      0@      "@       @      $@      $@      &@      "@      @      @       @        
�"
wd1*�"	   ��ɲ�   �u��?      A!�cRw�%�@)�3I"()T@2��{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ����?�ګ�;9��R���5�L�����]�����MZ��K���u��gr��R%������39W$:���X$�z��
�}�����4[_>������m!#���
�%W��ہkVl�p�w`f���n�=�.^ol�cR�k�e>:�AC)8g>�i����v>E'�/��x>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>R%�����>�u��gr�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?�������:�              @      3@     �G@      U@     �Y@     `d@     @k@     @r@     �y@     ��@      �@     ̗@     ��@     �@     ��@     X�@    ��@     t�@    �q�@     ��@     ��@     ��@    ��@    �L�@    ��@     :�@     \�@     ��@     �@     ��@     ͵@     ��@     ��@     ��@     Z�@     �@     ��@     �@     ��@      �@     ��@     x�@     ��@     ��@     $�@     ��@     �@     �@     ��@     x�@     p�@     0�@     �@     0�@     ��@     ��@      @     0{@      {@     �v@      x@     0u@     �q@     �q@     @n@     @j@     �f@     �f@     �b@      c@     �_@     �]@     �Y@     @R@     �Z@     �S@     �R@      P@      K@      D@     �P@      B@      B@     �I@      9@      7@      6@     �A@     �A@      3@      1@      0@      3@      @      "@      (@      ,@       @      @       @      @      "@      @      @      *@       @      @      @      @       @      @      �?       @      �?       @      @      �?      @               @       @      �?               @      �?      @              �?      �?              �?              �?      �?              �?              �?              �?      �?              �?              �?              @              @      �?      �?       @       @       @      @      @      �?      @      �?      @      @      $@      .@      &@      @      @      @      @      &@      ,@      0@       @       @      ,@      0@      ,@      5@      8@      =@      1@      =@      @@     �E@      H@     �E@     �J@     �G@     �P@      N@     �T@      R@      S@     �V@     �[@     �[@     @_@     @_@     �c@     `g@     @f@     �h@     �k@     �p@      r@     s@     �q@      s@     �y@     0y@     �z@     P~@     ��@     ��@     ��@     ȇ@     �@     H�@     ��@     P�@     Ԓ@     ,�@     L�@     ��@     �@     D�@     ��@     H�@     
�@     ��@     ܧ@     ��@     �@     z�@     �@     �@     ��@     #�@     6�@     �@     �@     �@     !�@     ��@    ���@    ��@     ��@     u�@     �@     �@     Q�@    �
�@     "�@    ���@     %�@     ��@     �@     ��@     <�@     Ȧ@     �@     L�@     ��@     X�@     ��@     �q@     �b@      C@      @        
�
wo*�	   `��    �ư?     ��@! `]&�7�)��&�?2�	����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��T7����5�i}1�O�ʗ�����Zr[v���ߊ4F��>})�l a�>����?f�ʜ�7
?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�������:�	              �?      �?      �?       @      @      ,@      $@      4@      5@      G@     �K@     �J@     �R@     @P@     �Q@     �R@     @S@     @R@     �L@     �Q@      J@     �K@     �J@     �D@     �J@      L@     �B@      :@     �E@      >@      >@      3@      <@      ?@      4@      1@      .@      (@      (@       @      &@      (@      @      @      *@      @      @      @      @      @      @       @      @      @       @      @      @       @      �?      �?       @      �?              �?              �?      �?       @              �?              @              �?              �?              �?      �?              �?              �?      �?      �?      @      @      �?       @       @      �?      @       @      @      @       @      @      @      @      @      @      @       @      $@      *@      @      &@      @      @      &@      1@      ,@      8@      6@      7@      5@      5@      9@      :@      A@     �@@     �D@     �@@      E@     �F@     �M@     �K@      J@     �C@      I@     �P@      N@     �G@     �A@     �A@      H@      :@      >@      1@      ,@      $@      @      @      @       @              �?               @        
�
bc1*�	   `�gz�   ��{�?      @@!  ��s��?)�T_��?2po��5sz�*QH�x�&b՞
�u����&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�/��?�uS��a�?���g��?I���?�������:p              �?      �?               @       @      $@      *@      �?              �?              �?        
�
bd1*�	    n�x�   ��7�?      P@!  `���?)h��录?2�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp�Tw��Nof?P}���h?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�������:�              7@      @       @      @       @              �?              �?      �?      �?              @      �?      @      @      @      @      �?      @        
�
bo*�	   @�3��   ���?     �E@!   ���տ)0.� R��?2�I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��>	� �����T}�*QH�x�&b՞
�u���ڋ��vV�R9��l�P�`?���%��b?5Ucv0ed?Tw��Nof?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?�"�uԖ?}Y�4j�?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?�������:�              �?              �?      �?       @       @      @      @      @      @               @              �?      �?              �?              �?              �?              �?              �?              �?              �?               @               @              �?       @      �?       @      �?        3ŕ|�&      a �E	��:�o�AB*�M

loss?�[@

accuracy��=
�
wc1*�	    qI��   `@D�?      r@!  iF���?)^�+ulƧ?2��uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A��5�i}1���d�r������~�f^��`{���VlQ.?��bȬ�0?��82?�u�w74?��%>��:?d�\D�X=?�T���C?a�$��{E?
����G?�qU���I?k�1^�sO?nK���LQ?�lDZrS?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?              �?      �?       @      �?      @      �?      @      @      @       @              @      @      @      @      @       @      @       @      @      @      @              @      @       @      @               @               @       @       @      @               @              �?              �?               @              �?              �?              �?              �?              �?              �?      �?       @              �?      �?              �?      @      �?       @      �?      @      @       @       @      @              @      @       @      @      @      @      @       @       @      $@      @      *@      *@      @      "@      $@      (@      &@      @      @      @        
�"
wd1*�"	    |7��    #�?      A!���~��@)�4��T@2��{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���X$�z��
�}�����4[_>������m!#���
�%W��E'�/��x��i����v���x��U>Fixі�W>�4[_>��>
�}���>X$�z�>�u��gr�>�MZ��K�>��|�~�>���]���>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?�������:�              @      ;@     �K@     @W@      _@     �f@     `m@     0t@     P{@     �@     @�@     ��@     �@     ^�@      �@     ��@     [�@     ��@     ��@    ��@    ��@     ��@     ��@    �h�@    ��@    �0�@     ��@     ��@     ��@     �@     E�@     Y�@     ��@     б@     ��@     b�@     Ȫ@     &�@     ��@     8�@     �@     $�@     ��@     L�@     �@     ��@     ��@     ��@     ��@     ؎@     `�@     h�@     ��@     ��@     ��@     (�@     @~@      {@     �{@     0v@     �w@     �s@     �p@     Pq@      n@     �i@     �g@     `e@     �a@     �_@     �_@     @]@     @Y@      U@     @X@      V@     @T@     �M@     �L@     �O@      N@      C@      D@     �D@      :@      =@      @@      ;@      B@      6@      4@      5@      .@      &@      .@      &@      *@      (@      @      @      @      @       @      $@      @      "@      @       @      �?      @      @              @      @      @      @      �?       @      �?               @              �?              �?      �?       @      �?               @      �?              �?              �?              �?              �?      �?              �?      @      @               @       @      �?      �?      @      @       @      @      �?      @      �?       @      @      @      "@      "@      @      "@      @      @      @      ,@       @      2@      *@      $@      .@      ,@      $@      1@      7@      8@      :@      2@     �A@     �A@      G@      A@     �E@     �J@      G@     �N@      J@      R@      S@      Q@     �U@     �Y@      W@      a@     �`@     @c@     �f@      e@      g@     `j@      p@     `q@     Ps@     �r@     t@      y@     �x@     �{@     @     �@     �@     ��@     x�@     �@     8�@     (�@     p�@     H�@     Ԕ@     \�@     �@     (�@     |�@     "�@     ¡@     �@     �@     ��@     j�@      �@     D�@     �@     ��@     ��@     ��@     �@     }�@     X�@     ��@     �@    ���@     -�@    ���@     ��@     q�@    ���@     ��@     ��@     ��@     �@    �V�@      �@     2�@     ��@     ��@     ܭ@     H�@     4�@     0�@     t�@     H�@     h�@     `u@     �f@     �O@      (@        
�
wo*�	   @�\��   ��#�?     ��@! x��53�)�6ʾ_�?2�	��]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ�x?�x��>h�'��O�ʗ�����Zr[v��a�Ϭ(�>8K�ߝ�>�ߊ4F��>})�l a�>�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�������:�	              �?              �?       @       @      @      .@      $@      2@      6@     �G@      K@     �J@     @S@     �P@     �Q@     �R@     �R@      S@      I@      S@     �J@     �I@      J@     �F@     �K@     �I@     �C@      =@      D@      =@      >@      3@      >@      <@      3@      2@      0@      .@      "@       @      "@      (@       @       @      (@       @      @      @       @      @      @      @      @      �?      @      @      @      �?      �?       @              �?      �?      �?      �?      �?       @              �?              @              �?              �?              �?      �?              �?               @      �?      �?       @      @      �?      �?              �?               @      @      @      @      @      @      @      @      @      @       @       @      (@      @      $@      @      $@      &@      2@      *@      6@      6@      8@      4@      6@      8@      :@      B@      =@      F@      ?@      E@     �F@      P@      I@     �I@      C@      J@      P@     �N@      I@     �@@     �A@     �H@      :@      >@      0@      .@       @      @      @       @      @              �?               @        
�
bc1*�	    �hz�   @&;�?      @@!  �����?)��Ȯ�R�?2po��5sz�*QH�x�&b՞
�u����&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�uS��a�?`��a�8�?���g��?I���?�������:p              �?      �?              �?      @      $@      (@       @              �?              �?        
�
bd1*�	   �q�x�   �Ar�?      P@!  @M���?)�	�bd�?2�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp�5Ucv0ed?Tw��Nof?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?�Rc�ݒ?^�S���?