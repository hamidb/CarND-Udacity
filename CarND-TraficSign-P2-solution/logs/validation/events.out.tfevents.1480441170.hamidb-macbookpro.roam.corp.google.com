       �K"	  �To�Abrain.Event:2��u��      B��	��To�A"��
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
: "on5�\#      $�@�	�ua_o�A
*�F

lossG�c@

accuracy��D=
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
\��   �3t�?     �E@!   us��)���{�p?2P-Ա�L�����J�\������=���>	� ��o��5sz?���T}?����=��?���J�\�?-Ա�L�?�������:P              9@              �?              �?              @      *@        ���j$      y�6�	e�Lgo�A*�H

loss�/a@

accuracy�Ga=
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
�u?*QH�x?o��5sz?����=��?���J�\�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�������:�              �?      @      $@      @      �?               @              �?              �?              �?              �?       @              �?               @       @       @      @      �?        }���%       �	�\oo�A*�K

lossf�`@

accuracy�lg=
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
�u?����=��?���J�\�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�������:�              �?              �?      $@      @      @      �?               @              �?              �?              �?              �?              �?              �?              �?              �?              �?       @      �?              @      @      �?        ~�q<%      X�	z��vo�A(*�J

losss`@

accuracy��o=
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
�u��l�P�`?���%��b?P}���h?ߤ�(g%k?����=��?���J�\�?-Ա�L�?eiS�m�?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�������:�              �?              �?      �?      @      $@       @               @       @               @      �?      �?              �?              �?              �?              �?              �?              �?              �?      �?      �?               @       @       @      �?        ��5�%      Z\H�	�<~o�A2*�K

loss0R_@

accuracy  �=
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
�u?o��5sz?���T}?>	� �?����=��?�7c_XY�?�#�h/�?���&�?^�S���?�"�uԖ?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?�������:�              �?              �?      �?      @      @      @      @              �?      �?       @       @              �?              �?              �?              �?              �?              �?              �?              �?      �?               @              @      �?       @      �?        �*�$&      >:�	r�؅o�A<*�K

loss�]@

accuracy��=
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
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ���[���FF�G ���Zr[v��I��P=��pz�w�7���ߊ4F��h���`iD*L��>E��a�W�>��(���>a�Ϭ(�>�h���`�>�ߊ4F��>6�]��?����?�T7��?�vV�R9