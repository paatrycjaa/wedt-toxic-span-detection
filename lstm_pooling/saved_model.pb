ЉС&
ЭЃ
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
О
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.12v2.3.0-54-gfcc4b966f18ђђ$

embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:ЌЌ*%
shared_nameembedding/embeddings

(embedding/embeddings/Read/ReadVariableOpReadVariableOpembedding/embeddings*!
_output_shapes
:ЌЌ*
dtype0

dense_final/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*#
shared_namedense_final/kernel
y
&dense_final/kernel/Read/ReadVariableOpReadVariableOpdense_final/kernel*
_output_shapes

:d*
dtype0
x
dense_final/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namedense_final/bias
q
$dense_final/bias/Read/ReadVariableOpReadVariableOpdense_final/bias*
_output_shapes
:*
dtype0
l
RMSprop/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameRMSprop/iter
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
_output_shapes
: *
dtype0	
n
RMSprop/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/decay
g
!RMSprop/decay/Read/ReadVariableOpReadVariableOpRMSprop/decay*
_output_shapes
: *
dtype0
~
RMSprop/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameRMSprop/learning_rate
w
)RMSprop/learning_rate/Read/ReadVariableOpReadVariableOpRMSprop/learning_rate*
_output_shapes
: *
dtype0
t
RMSprop/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameRMSprop/momentum
m
$RMSprop/momentum/Read/ReadVariableOpReadVariableOpRMSprop/momentum*
_output_shapes
: *
dtype0
j
RMSprop/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/rho
c
RMSprop/rho/Read/ReadVariableOpReadVariableOpRMSprop/rho*
_output_shapes
: *
dtype0

lstm/lstm_cell_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ќ*(
shared_namelstm/lstm_cell_4/kernel

+lstm/lstm_cell_4/kernel/Read/ReadVariableOpReadVariableOplstm/lstm_cell_4/kernel* 
_output_shapes
:
Ќ*
dtype0

!lstm/lstm_cell_4/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d*2
shared_name#!lstm/lstm_cell_4/recurrent_kernel

5lstm/lstm_cell_4/recurrent_kernel/Read/ReadVariableOpReadVariableOp!lstm/lstm_cell_4/recurrent_kernel*
_output_shapes
:	d*
dtype0

lstm/lstm_cell_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namelstm/lstm_cell_4/bias
|
)lstm/lstm_cell_4/bias/Read/ReadVariableOpReadVariableOplstm/lstm_cell_4/bias*
_output_shapes	
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

 RMSprop/embedding/embeddings/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:ЌЌ*1
shared_name" RMSprop/embedding/embeddings/rms

4RMSprop/embedding/embeddings/rms/Read/ReadVariableOpReadVariableOp RMSprop/embedding/embeddings/rms*!
_output_shapes
:ЌЌ*
dtype0

RMSprop/dense_final/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*/
shared_name RMSprop/dense_final/kernel/rms

2RMSprop/dense_final/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_final/kernel/rms*
_output_shapes

:d*
dtype0

RMSprop/dense_final/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameRMSprop/dense_final/bias/rms

0RMSprop/dense_final/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_final/bias/rms*
_output_shapes
:*
dtype0
Є
#RMSprop/lstm/lstm_cell_4/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ќ*4
shared_name%#RMSprop/lstm/lstm_cell_4/kernel/rms

7RMSprop/lstm/lstm_cell_4/kernel/rms/Read/ReadVariableOpReadVariableOp#RMSprop/lstm/lstm_cell_4/kernel/rms* 
_output_shapes
:
Ќ*
dtype0
З
-RMSprop/lstm/lstm_cell_4/recurrent_kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d*>
shared_name/-RMSprop/lstm/lstm_cell_4/recurrent_kernel/rms
А
ARMSprop/lstm/lstm_cell_4/recurrent_kernel/rms/Read/ReadVariableOpReadVariableOp-RMSprop/lstm/lstm_cell_4/recurrent_kernel/rms*
_output_shapes
:	d*
dtype0

!RMSprop/lstm/lstm_cell_4/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!RMSprop/lstm/lstm_cell_4/bias/rms

5RMSprop/lstm/lstm_cell_4/bias/rms/Read/ReadVariableOpReadVariableOp!RMSprop/lstm/lstm_cell_4/bias/rms*
_output_shapes	
:*
dtype0

NoOpNoOp
Б%
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ь$
valueт$Bп$ Bи$
ѓ
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
	optimizer
regularization_losses
trainable_variables
	variables
		keras_api


signatures
b

embeddings
regularization_losses
trainable_variables
	variables
	keras_api
l
cell

state_spec
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api

 iter
	!decay
"learning_rate
#momentum
$rho	rmsV	rmsW	rmsX	%rmsY	&rmsZ	'rms[
 
*
0
%1
&2
'3
4
5
*
0
%1
&2
'3
4
5
­
regularization_losses
(metrics
)layer_regularization_losses
*non_trainable_variables

+layers
trainable_variables
	variables
,layer_metrics
 
db
VARIABLE_VALUEembedding/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
­
regularization_losses
-metrics
.layer_regularization_losses
/non_trainable_variables

0layers
trainable_variables
	variables
1layer_metrics
~

%kernel
&recurrent_kernel
'bias
2regularization_losses
3trainable_variables
4	variables
5	keras_api
 
 

%0
&1
'2

%0
&1
'2
Й

6states
regularization_losses
7metrics
8layer_regularization_losses
9non_trainable_variables

:layers
trainable_variables
	variables
;layer_metrics
 
 
 
­
regularization_losses
<metrics
=layer_regularization_losses
>non_trainable_variables

?layers
trainable_variables
	variables
@layer_metrics
^\
VARIABLE_VALUEdense_final/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEdense_final/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
regularization_losses
Ametrics
Blayer_regularization_losses
Cnon_trainable_variables

Dlayers
trainable_variables
	variables
Elayer_metrics
KI
VARIABLE_VALUERMSprop/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUERMSprop/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUERMSprop/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUERMSprop/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUERMSprop/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUElstm/lstm_cell_4/kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE!lstm/lstm_cell_4/recurrent_kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUElstm/lstm_cell_4/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE

F0
G1
 
 

0
1
2
3
 
 
 
 
 
 
 

%0
&1
'2

%0
&1
'2
­
2regularization_losses
Hmetrics
Ilayer_regularization_losses
Jnon_trainable_variables

Klayers
3trainable_variables
4	variables
Llayer_metrics
 
 
 
 

0
 
 
 
 
 
 
 
 
 
 
 
4
	Mtotal
	Ncount
O	variables
P	keras_api
D
	Qtotal
	Rcount
S
_fn_kwargs
T	variables
U	keras_api
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

M0
N1

O	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

Q0
R1

T	variables

VARIABLE_VALUE RMSprop/embedding/embeddings/rmsXlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/dense_final/kernel/rmsTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/dense_final/bias/rmsRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#RMSprop/lstm/lstm_cell_4/kernel/rmsNtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE-RMSprop/lstm/lstm_cell_4/recurrent_kernel/rmsNtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!RMSprop/lstm/lstm_cell_4/bias/rmsNtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

serving_default_embedding_inputPlaceholder*'
_output_shapes
:џџџџџџџџџ(*
dtype0*
shape:џџџџџџџџџ(
д
StatefulPartitionedCallStatefulPartitionedCallserving_default_embedding_inputembedding/embeddingslstm/lstm_cell_4/kernellstm/lstm_cell_4/bias!lstm/lstm_cell_4/recurrent_kerneldense_final/kerneldense_final/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_32868
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ќ	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename(embedding/embeddings/Read/ReadVariableOp&dense_final/kernel/Read/ReadVariableOp$dense_final/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOp+lstm/lstm_cell_4/kernel/Read/ReadVariableOp5lstm/lstm_cell_4/recurrent_kernel/Read/ReadVariableOp)lstm/lstm_cell_4/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp4RMSprop/embedding/embeddings/rms/Read/ReadVariableOp2RMSprop/dense_final/kernel/rms/Read/ReadVariableOp0RMSprop/dense_final/bias/rms/Read/ReadVariableOp7RMSprop/lstm/lstm_cell_4/kernel/rms/Read/ReadVariableOpARMSprop/lstm/lstm_cell_4/recurrent_kernel/rms/Read/ReadVariableOp5RMSprop/lstm/lstm_cell_4/bias/rms/Read/ReadVariableOpConst*"
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_35293

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding/embeddingsdense_final/kerneldense_final/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rholstm/lstm_cell_4/kernel!lstm/lstm_cell_4/recurrent_kernellstm/lstm_cell_4/biastotalcounttotal_1count_1 RMSprop/embedding/embeddings/rmsRMSprop/dense_final/kernel/rmsRMSprop/dense_final/bias/rms#RMSprop/lstm/lstm_cell_4/kernel/rms-RMSprop/lstm/lstm_cell_4/recurrent_kernel/rms!RMSprop/lstm/lstm_cell_4/bias/rms*!
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_35366Ў$
К
я
?__inference_lstm_layer_call_and_return_conditional_losses_32667

inputs-
)lstm_cell_4_split_readvariableop_resource/
+lstm_cell_4_split_1_readvariableop_resource'
#lstm_cell_4_readvariableop_resource
identityЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:(џџџџџџџџџЌ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2§
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
shrink_axis_mask2
strided_slice_2
lstm_cell_4/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell_4/ones_like/Shape
lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_4/ones_like/ConstЕ
lstm_cell_4/ones_likeFill$lstm_cell_4/ones_like/Shape:output:0$lstm_cell_4/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/ones_like|
lstm_cell_4/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_4/ones_like_1/Shape
lstm_cell_4/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_4/ones_like_1/ConstМ
lstm_cell_4/ones_like_1Fill&lstm_cell_4/ones_like_1/Shape:output:0&lstm_cell_4/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/ones_like_1
lstm_cell_4/mulMulstrided_slice_2:output:0lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/mul
lstm_cell_4/mul_1Mulstrided_slice_2:output:0lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/mul_1
lstm_cell_4/mul_2Mulstrided_slice_2:output:0lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/mul_2
lstm_cell_4/mul_3Mulstrided_slice_2:output:0lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/mul_3h
lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/Const|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dimА
 lstm_cell_4/split/ReadVariableOpReadVariableOp)lstm_cell_4_split_readvariableop_resource* 
_output_shapes
:
Ќ*
dtype02"
 lstm_cell_4/split/ReadVariableOpл
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0(lstm_cell_4/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	Ќd:	Ќd:	Ќd:	Ќd*
	num_split2
lstm_cell_4/split
lstm_cell_4/MatMulMatMullstm_cell_4/mul:z:0lstm_cell_4/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul
lstm_cell_4/MatMul_1MatMullstm_cell_4/mul_1:z:0lstm_cell_4/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul_1
lstm_cell_4/MatMul_2MatMullstm_cell_4/mul_2:z:0lstm_cell_4/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul_2
lstm_cell_4/MatMul_3MatMullstm_cell_4/mul_3:z:0lstm_cell_4/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul_3l
lstm_cell_4/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/Const_1
lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_4/split_1/split_dimБ
"lstm_cell_4/split_1/ReadVariableOpReadVariableOp+lstm_cell_4_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02$
"lstm_cell_4/split_1/ReadVariableOpЯ
lstm_cell_4/split_1Split&lstm_cell_4/split_1/split_dim:output:0*lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:d:d:d:d*
	num_split2
lstm_cell_4/split_1Ѓ
lstm_cell_4/BiasAddBiasAddlstm_cell_4/MatMul:product:0lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/BiasAddЉ
lstm_cell_4/BiasAdd_1BiasAddlstm_cell_4/MatMul_1:product:0lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/BiasAdd_1Љ
lstm_cell_4/BiasAdd_2BiasAddlstm_cell_4/MatMul_2:product:0lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/BiasAdd_2Љ
lstm_cell_4/BiasAdd_3BiasAddlstm_cell_4/MatMul_3:product:0lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/BiasAdd_3
lstm_cell_4/mul_4Mulzeros:output:0 lstm_cell_4/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/mul_4
lstm_cell_4/mul_5Mulzeros:output:0 lstm_cell_4/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/mul_5
lstm_cell_4/mul_6Mulzeros:output:0 lstm_cell_4/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/mul_6
lstm_cell_4/mul_7Mulzeros:output:0 lstm_cell_4/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/mul_7
lstm_cell_4/ReadVariableOpReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	d*
dtype02
lstm_cell_4/ReadVariableOp
lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell_4/strided_slice/stack
!lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   2#
!lstm_cell_4/strided_slice/stack_1
!lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell_4/strided_slice/stack_2Ф
lstm_cell_4/strided_sliceStridedSlice"lstm_cell_4/ReadVariableOp:value:0(lstm_cell_4/strided_slice/stack:output:0*lstm_cell_4/strided_slice/stack_1:output:0*lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2
lstm_cell_4/strided_sliceЃ
lstm_cell_4/MatMul_4MatMullstm_cell_4/mul_4:z:0"lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul_4
lstm_cell_4/addAddV2lstm_cell_4/BiasAdd:output:0lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/add|
lstm_cell_4/SigmoidSigmoidlstm_cell_4/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/SigmoidЁ
lstm_cell_4/ReadVariableOp_1ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	d*
dtype02
lstm_cell_4/ReadVariableOp_1
!lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   2#
!lstm_cell_4/strided_slice_1/stack
#lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Ш   2%
#lstm_cell_4/strided_slice_1/stack_1
#lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_4/strided_slice_1/stack_2а
lstm_cell_4/strided_slice_1StridedSlice$lstm_cell_4/ReadVariableOp_1:value:0*lstm_cell_4/strided_slice_1/stack:output:0,lstm_cell_4/strided_slice_1/stack_1:output:0,lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2
lstm_cell_4/strided_slice_1Ѕ
lstm_cell_4/MatMul_5MatMullstm_cell_4/mul_5:z:0$lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul_5Ё
lstm_cell_4/add_1AddV2lstm_cell_4/BiasAdd_1:output:0lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/add_1
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/Sigmoid_1
lstm_cell_4/mul_8Mullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/mul_8Ё
lstm_cell_4/ReadVariableOp_2ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	d*
dtype02
lstm_cell_4/ReadVariableOp_2
!lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    Ш   2#
!lstm_cell_4/strided_slice_2/stack
#lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  2%
#lstm_cell_4/strided_slice_2/stack_1
#lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_4/strided_slice_2/stack_2а
lstm_cell_4/strided_slice_2StridedSlice$lstm_cell_4/ReadVariableOp_2:value:0*lstm_cell_4/strided_slice_2/stack:output:0,lstm_cell_4/strided_slice_2/stack_1:output:0,lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2
lstm_cell_4/strided_slice_2Ѕ
lstm_cell_4/MatMul_6MatMullstm_cell_4/mul_6:z:0$lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul_6Ё
lstm_cell_4/add_2AddV2lstm_cell_4/BiasAdd_2:output:0lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/add_2u
lstm_cell_4/TanhTanhlstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/Tanh
lstm_cell_4/mul_9Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/mul_9
lstm_cell_4/add_3AddV2lstm_cell_4/mul_8:z:0lstm_cell_4/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/add_3Ё
lstm_cell_4/ReadVariableOp_3ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	d*
dtype02
lstm_cell_4/ReadVariableOp_3
!lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  2#
!lstm_cell_4/strided_slice_3/stack
#lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_4/strided_slice_3/stack_1
#lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_4/strided_slice_3/stack_2а
lstm_cell_4/strided_slice_3StridedSlice$lstm_cell_4/ReadVariableOp_3:value:0*lstm_cell_4/strided_slice_3/stack:output:0,lstm_cell_4/strided_slice_3/stack_1:output:0,lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2
lstm_cell_4/strided_slice_3Ѕ
lstm_cell_4/MatMul_7MatMullstm_cell_4/mul_7:z:0$lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul_7Ё
lstm_cell_4/add_4AddV2lstm_cell_4/BiasAdd_3:output:0lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/add_4
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Tanh_1Tanhlstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/Tanh_1
lstm_cell_4/mul_10Mullstm_cell_4/Sigmoid_2:y:0lstm_cell_4/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/mul_10
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterп
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_4_split_readvariableop_resource+lstm_cell_4_split_1_readvariableop_resource#lstm_cell_4_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_32531*
condR
while_cond_32530*K
output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:(џџџџџџџџџd*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЅ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ(d2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:џџџџџџџџџ(Ќ:::2
whilewhile:T P
,
_output_shapes
:џџџџџџџџџ(Ќ
 
_user_specified_nameinputs


$__inference_lstm_layer_call_fn_34243
inputs_0
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_319952
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:џџџџџџџџџџџџџџџџџџЌ:::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ
"
_user_specified_name
inputs/0
Э

E__inference_sequential_layer_call_and_return_conditional_losses_33264

inputs$
 embedding_embedding_lookup_328722
.lstm_lstm_cell_4_split_readvariableop_resource4
0lstm_lstm_cell_4_split_1_readvariableop_resource,
(lstm_lstm_cell_4_readvariableop_resource.
*dense_final_matmul_readvariableop_resource/
+dense_final_biasadd_readvariableop_resource
identityЂ
lstm/whileq
embedding/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ(2
embedding/Cast
embedding/embedding_lookupResourceGather embedding_embedding_lookup_32872embedding/Cast:y:0*
Tindices0*3
_class)
'%loc:@embedding/embedding_lookup/32872*,
_output_shapes
:џџџџџџџџџ(Ќ*
dtype02
embedding/embedding_lookupч
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*3
_class)
'%loc:@embedding/embedding_lookup/32872*,
_output_shapes
:џџџџџџџџџ(Ќ2%
#embedding/embedding_lookup/IdentityП
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:џџџџџџџџџ(Ќ2'
%embedding/embedding_lookup/Identity_1v

lstm/ShapeShape.embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2

lstm/Shape~
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice/stack
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_1
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_2
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slicef
lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
lstm/zeros/mul/y
lstm/zeros/mulMullstm/strided_slice:output:0lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/muli
lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm/zeros/Less/y{
lstm/zeros/LessLesslstm/zeros/mul:z:0lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/Lessl
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
lstm/zeros/packed/1
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros/packedi
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros/Const

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2

lstm/zerosj
lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
lstm/zeros_1/mul/y
lstm/zeros_1/mulMullstm/strided_slice:output:0lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/mulm
lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm/zeros_1/Less/y
lstm/zeros_1/LessLesslstm/zeros_1/mul:z:0lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/Lessp
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
lstm/zeros_1/packed/1
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros_1/packedm
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros_1/Const
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/zeros_1
lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose/permВ
lstm/transpose	Transpose.embedding/embedding_lookup/Identity_1:output:0lstm/transpose/perm:output:0*
T0*,
_output_shapes
:(џџџџџџџџџЌ2
lstm/transpose^
lstm/Shape_1Shapelstm/transpose:y:0*
T0*
_output_shapes
:2
lstm/Shape_1
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_1/stack
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_1
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_2
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slice_1
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2"
 lstm/TensorArrayV2/element_shapeЦ
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2Щ
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  2<
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shape
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02.
,lstm/TensorArrayUnstack/TensorListFromTensor
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_2/stack
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_1
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_2
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
shrink_axis_mask2
lstm/strided_slice_2
 lstm/lstm_cell_4/ones_like/ShapeShapelstm/strided_slice_2:output:0*
T0*
_output_shapes
:2"
 lstm/lstm_cell_4/ones_like/Shape
 lstm/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2"
 lstm/lstm_cell_4/ones_like/ConstЩ
lstm/lstm_cell_4/ones_likeFill)lstm/lstm_cell_4/ones_like/Shape:output:0)lstm/lstm_cell_4/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm/lstm_cell_4/ones_like
lstm/lstm_cell_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2 
lstm/lstm_cell_4/dropout/ConstФ
lstm/lstm_cell_4/dropout/MulMul#lstm/lstm_cell_4/ones_like:output:0'lstm/lstm_cell_4/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm/lstm_cell_4/dropout/Mul
lstm/lstm_cell_4/dropout/ShapeShape#lstm/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2 
lstm/lstm_cell_4/dropout/Shape
5lstm/lstm_cell_4/dropout/random_uniform/RandomUniformRandomUniform'lstm/lstm_cell_4/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*
seedБџх)*
seed2Ца27
5lstm/lstm_cell_4/dropout/random_uniform/RandomUniform
'lstm/lstm_cell_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2)
'lstm/lstm_cell_4/dropout/GreaterEqual/y
%lstm/lstm_cell_4/dropout/GreaterEqualGreaterEqual>lstm/lstm_cell_4/dropout/random_uniform/RandomUniform:output:00lstm/lstm_cell_4/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2'
%lstm/lstm_cell_4/dropout/GreaterEqualГ
lstm/lstm_cell_4/dropout/CastCast)lstm/lstm_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
lstm/lstm_cell_4/dropout/CastП
lstm/lstm_cell_4/dropout/Mul_1Mul lstm/lstm_cell_4/dropout/Mul:z:0!lstm/lstm_cell_4/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2 
lstm/lstm_cell_4/dropout/Mul_1
 lstm/lstm_cell_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2"
 lstm/lstm_cell_4/dropout_1/ConstЪ
lstm/lstm_cell_4/dropout_1/MulMul#lstm/lstm_cell_4/ones_like:output:0)lstm/lstm_cell_4/dropout_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2 
lstm/lstm_cell_4/dropout_1/Mul
 lstm/lstm_cell_4/dropout_1/ShapeShape#lstm/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2"
 lstm/lstm_cell_4/dropout_1/Shape
7lstm/lstm_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform)lstm/lstm_cell_4/dropout_1/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*
seedБџх)*
seed2ёй29
7lstm/lstm_cell_4/dropout_1/random_uniform/RandomUniform
)lstm/lstm_cell_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2+
)lstm/lstm_cell_4/dropout_1/GreaterEqual/y
'lstm/lstm_cell_4/dropout_1/GreaterEqualGreaterEqual@lstm/lstm_cell_4/dropout_1/random_uniform/RandomUniform:output:02lstm/lstm_cell_4/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2)
'lstm/lstm_cell_4/dropout_1/GreaterEqualЙ
lstm/lstm_cell_4/dropout_1/CastCast+lstm/lstm_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2!
lstm/lstm_cell_4/dropout_1/CastЧ
 lstm/lstm_cell_4/dropout_1/Mul_1Mul"lstm/lstm_cell_4/dropout_1/Mul:z:0#lstm/lstm_cell_4/dropout_1/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2"
 lstm/lstm_cell_4/dropout_1/Mul_1
 lstm/lstm_cell_4/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2"
 lstm/lstm_cell_4/dropout_2/ConstЪ
lstm/lstm_cell_4/dropout_2/MulMul#lstm/lstm_cell_4/ones_like:output:0)lstm/lstm_cell_4/dropout_2/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2 
lstm/lstm_cell_4/dropout_2/Mul
 lstm/lstm_cell_4/dropout_2/ShapeShape#lstm/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2"
 lstm/lstm_cell_4/dropout_2/Shape
7lstm/lstm_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform)lstm/lstm_cell_4/dropout_2/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*
seedБџх)*
seed2бвА29
7lstm/lstm_cell_4/dropout_2/random_uniform/RandomUniform
)lstm/lstm_cell_4/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2+
)lstm/lstm_cell_4/dropout_2/GreaterEqual/y
'lstm/lstm_cell_4/dropout_2/GreaterEqualGreaterEqual@lstm/lstm_cell_4/dropout_2/random_uniform/RandomUniform:output:02lstm/lstm_cell_4/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2)
'lstm/lstm_cell_4/dropout_2/GreaterEqualЙ
lstm/lstm_cell_4/dropout_2/CastCast+lstm/lstm_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2!
lstm/lstm_cell_4/dropout_2/CastЧ
 lstm/lstm_cell_4/dropout_2/Mul_1Mul"lstm/lstm_cell_4/dropout_2/Mul:z:0#lstm/lstm_cell_4/dropout_2/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2"
 lstm/lstm_cell_4/dropout_2/Mul_1
 lstm/lstm_cell_4/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2"
 lstm/lstm_cell_4/dropout_3/ConstЪ
lstm/lstm_cell_4/dropout_3/MulMul#lstm/lstm_cell_4/ones_like:output:0)lstm/lstm_cell_4/dropout_3/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2 
lstm/lstm_cell_4/dropout_3/Mul
 lstm/lstm_cell_4/dropout_3/ShapeShape#lstm/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2"
 lstm/lstm_cell_4/dropout_3/Shape
7lstm/lstm_cell_4/dropout_3/random_uniform/RandomUniformRandomUniform)lstm/lstm_cell_4/dropout_3/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*
seedБџх)*
seed2ЏЪ29
7lstm/lstm_cell_4/dropout_3/random_uniform/RandomUniform
)lstm/lstm_cell_4/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2+
)lstm/lstm_cell_4/dropout_3/GreaterEqual/y
'lstm/lstm_cell_4/dropout_3/GreaterEqualGreaterEqual@lstm/lstm_cell_4/dropout_3/random_uniform/RandomUniform:output:02lstm/lstm_cell_4/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2)
'lstm/lstm_cell_4/dropout_3/GreaterEqualЙ
lstm/lstm_cell_4/dropout_3/CastCast+lstm/lstm_cell_4/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2!
lstm/lstm_cell_4/dropout_3/CastЧ
 lstm/lstm_cell_4/dropout_3/Mul_1Mul"lstm/lstm_cell_4/dropout_3/Mul:z:0#lstm/lstm_cell_4/dropout_3/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2"
 lstm/lstm_cell_4/dropout_3/Mul_1
"lstm/lstm_cell_4/ones_like_1/ShapeShapelstm/zeros:output:0*
T0*
_output_shapes
:2$
"lstm/lstm_cell_4/ones_like_1/Shape
"lstm/lstm_cell_4/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"lstm/lstm_cell_4/ones_like_1/Constа
lstm/lstm_cell_4/ones_like_1Fill+lstm/lstm_cell_4/ones_like_1/Shape:output:0+lstm/lstm_cell_4/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/ones_like_1
 lstm/lstm_cell_4/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2"
 lstm/lstm_cell_4/dropout_4/ConstЫ
lstm/lstm_cell_4/dropout_4/MulMul%lstm/lstm_cell_4/ones_like_1:output:0)lstm/lstm_cell_4/dropout_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2 
lstm/lstm_cell_4/dropout_4/Mul
 lstm/lstm_cell_4/dropout_4/ShapeShape%lstm/lstm_cell_4/ones_like_1:output:0*
T0*
_output_shapes
:2"
 lstm/lstm_cell_4/dropout_4/Shape
7lstm/lstm_cell_4/dropout_4/random_uniform/RandomUniformRandomUniform)lstm/lstm_cell_4/dropout_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџd*
dtype0*
seedБџх)*
seed2Щш29
7lstm/lstm_cell_4/dropout_4/random_uniform/RandomUniform
)lstm/lstm_cell_4/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2+
)lstm/lstm_cell_4/dropout_4/GreaterEqual/y
'lstm/lstm_cell_4/dropout_4/GreaterEqualGreaterEqual@lstm/lstm_cell_4/dropout_4/random_uniform/RandomUniform:output:02lstm/lstm_cell_4/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2)
'lstm/lstm_cell_4/dropout_4/GreaterEqualИ
lstm/lstm_cell_4/dropout_4/CastCast+lstm/lstm_cell_4/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd2!
lstm/lstm_cell_4/dropout_4/CastЦ
 lstm/lstm_cell_4/dropout_4/Mul_1Mul"lstm/lstm_cell_4/dropout_4/Mul:z:0#lstm/lstm_cell_4/dropout_4/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2"
 lstm/lstm_cell_4/dropout_4/Mul_1
 lstm/lstm_cell_4/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2"
 lstm/lstm_cell_4/dropout_5/ConstЫ
lstm/lstm_cell_4/dropout_5/MulMul%lstm/lstm_cell_4/ones_like_1:output:0)lstm/lstm_cell_4/dropout_5/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2 
lstm/lstm_cell_4/dropout_5/Mul
 lstm/lstm_cell_4/dropout_5/ShapeShape%lstm/lstm_cell_4/ones_like_1:output:0*
T0*
_output_shapes
:2"
 lstm/lstm_cell_4/dropout_5/Shape
7lstm/lstm_cell_4/dropout_5/random_uniform/RandomUniformRandomUniform)lstm/lstm_cell_4/dropout_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџd*
dtype0*
seedБџх)*
seed2њБЛ29
7lstm/lstm_cell_4/dropout_5/random_uniform/RandomUniform
)lstm/lstm_cell_4/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2+
)lstm/lstm_cell_4/dropout_5/GreaterEqual/y
'lstm/lstm_cell_4/dropout_5/GreaterEqualGreaterEqual@lstm/lstm_cell_4/dropout_5/random_uniform/RandomUniform:output:02lstm/lstm_cell_4/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2)
'lstm/lstm_cell_4/dropout_5/GreaterEqualИ
lstm/lstm_cell_4/dropout_5/CastCast+lstm/lstm_cell_4/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd2!
lstm/lstm_cell_4/dropout_5/CastЦ
 lstm/lstm_cell_4/dropout_5/Mul_1Mul"lstm/lstm_cell_4/dropout_5/Mul:z:0#lstm/lstm_cell_4/dropout_5/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2"
 lstm/lstm_cell_4/dropout_5/Mul_1
 lstm/lstm_cell_4/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2"
 lstm/lstm_cell_4/dropout_6/ConstЫ
lstm/lstm_cell_4/dropout_6/MulMul%lstm/lstm_cell_4/ones_like_1:output:0)lstm/lstm_cell_4/dropout_6/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2 
lstm/lstm_cell_4/dropout_6/Mul
 lstm/lstm_cell_4/dropout_6/ShapeShape%lstm/lstm_cell_4/ones_like_1:output:0*
T0*
_output_shapes
:2"
 lstm/lstm_cell_4/dropout_6/Shape
7lstm/lstm_cell_4/dropout_6/random_uniform/RandomUniformRandomUniform)lstm/lstm_cell_4/dropout_6/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџd*
dtype0*
seedБџх)*
seed2е29
7lstm/lstm_cell_4/dropout_6/random_uniform/RandomUniform
)lstm/lstm_cell_4/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2+
)lstm/lstm_cell_4/dropout_6/GreaterEqual/y
'lstm/lstm_cell_4/dropout_6/GreaterEqualGreaterEqual@lstm/lstm_cell_4/dropout_6/random_uniform/RandomUniform:output:02lstm/lstm_cell_4/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2)
'lstm/lstm_cell_4/dropout_6/GreaterEqualИ
lstm/lstm_cell_4/dropout_6/CastCast+lstm/lstm_cell_4/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd2!
lstm/lstm_cell_4/dropout_6/CastЦ
 lstm/lstm_cell_4/dropout_6/Mul_1Mul"lstm/lstm_cell_4/dropout_6/Mul:z:0#lstm/lstm_cell_4/dropout_6/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2"
 lstm/lstm_cell_4/dropout_6/Mul_1
 lstm/lstm_cell_4/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2"
 lstm/lstm_cell_4/dropout_7/ConstЫ
lstm/lstm_cell_4/dropout_7/MulMul%lstm/lstm_cell_4/ones_like_1:output:0)lstm/lstm_cell_4/dropout_7/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2 
lstm/lstm_cell_4/dropout_7/Mul
 lstm/lstm_cell_4/dropout_7/ShapeShape%lstm/lstm_cell_4/ones_like_1:output:0*
T0*
_output_shapes
:2"
 lstm/lstm_cell_4/dropout_7/Shape
7lstm/lstm_cell_4/dropout_7/random_uniform/RandomUniformRandomUniform)lstm/lstm_cell_4/dropout_7/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџd*
dtype0*
seedБџх)*
seed2щз29
7lstm/lstm_cell_4/dropout_7/random_uniform/RandomUniform
)lstm/lstm_cell_4/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2+
)lstm/lstm_cell_4/dropout_7/GreaterEqual/y
'lstm/lstm_cell_4/dropout_7/GreaterEqualGreaterEqual@lstm/lstm_cell_4/dropout_7/random_uniform/RandomUniform:output:02lstm/lstm_cell_4/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2)
'lstm/lstm_cell_4/dropout_7/GreaterEqualИ
lstm/lstm_cell_4/dropout_7/CastCast+lstm/lstm_cell_4/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd2!
lstm/lstm_cell_4/dropout_7/CastЦ
 lstm/lstm_cell_4/dropout_7/Mul_1Mul"lstm/lstm_cell_4/dropout_7/Mul:z:0#lstm/lstm_cell_4/dropout_7/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2"
 lstm/lstm_cell_4/dropout_7/Mul_1Љ
lstm/lstm_cell_4/mulMullstm/strided_slice_2:output:0"lstm/lstm_cell_4/dropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm/lstm_cell_4/mulЏ
lstm/lstm_cell_4/mul_1Mullstm/strided_slice_2:output:0$lstm/lstm_cell_4/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm/lstm_cell_4/mul_1Џ
lstm/lstm_cell_4/mul_2Mullstm/strided_slice_2:output:0$lstm/lstm_cell_4/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm/lstm_cell_4/mul_2Џ
lstm/lstm_cell_4/mul_3Mullstm/strided_slice_2:output:0$lstm/lstm_cell_4/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm/lstm_cell_4/mul_3r
lstm/lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/lstm_cell_4/Const
 lstm/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 lstm/lstm_cell_4/split/split_dimП
%lstm/lstm_cell_4/split/ReadVariableOpReadVariableOp.lstm_lstm_cell_4_split_readvariableop_resource* 
_output_shapes
:
Ќ*
dtype02'
%lstm/lstm_cell_4/split/ReadVariableOpя
lstm/lstm_cell_4/splitSplit)lstm/lstm_cell_4/split/split_dim:output:0-lstm/lstm_cell_4/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	Ќd:	Ќd:	Ќd:	Ќd*
	num_split2
lstm/lstm_cell_4/splitЉ
lstm/lstm_cell_4/MatMulMatMullstm/lstm_cell_4/mul:z:0lstm/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/MatMulЏ
lstm/lstm_cell_4/MatMul_1MatMullstm/lstm_cell_4/mul_1:z:0lstm/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/MatMul_1Џ
lstm/lstm_cell_4/MatMul_2MatMullstm/lstm_cell_4/mul_2:z:0lstm/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/MatMul_2Џ
lstm/lstm_cell_4/MatMul_3MatMullstm/lstm_cell_4/mul_3:z:0lstm/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/MatMul_3v
lstm/lstm_cell_4/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm/lstm_cell_4/Const_1
"lstm/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"lstm/lstm_cell_4/split_1/split_dimР
'lstm/lstm_cell_4/split_1/ReadVariableOpReadVariableOp0lstm_lstm_cell_4_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02)
'lstm/lstm_cell_4/split_1/ReadVariableOpу
lstm/lstm_cell_4/split_1Split+lstm/lstm_cell_4/split_1/split_dim:output:0/lstm/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:d:d:d:d*
	num_split2
lstm/lstm_cell_4/split_1З
lstm/lstm_cell_4/BiasAddBiasAdd!lstm/lstm_cell_4/MatMul:product:0!lstm/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/BiasAddН
lstm/lstm_cell_4/BiasAdd_1BiasAdd#lstm/lstm_cell_4/MatMul_1:product:0!lstm/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/BiasAdd_1Н
lstm/lstm_cell_4/BiasAdd_2BiasAdd#lstm/lstm_cell_4/MatMul_2:product:0!lstm/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/BiasAdd_2Н
lstm/lstm_cell_4/BiasAdd_3BiasAdd#lstm/lstm_cell_4/MatMul_3:product:0!lstm/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/BiasAdd_3Є
lstm/lstm_cell_4/mul_4Mullstm/zeros:output:0$lstm/lstm_cell_4/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/mul_4Є
lstm/lstm_cell_4/mul_5Mullstm/zeros:output:0$lstm/lstm_cell_4/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/mul_5Є
lstm/lstm_cell_4/mul_6Mullstm/zeros:output:0$lstm/lstm_cell_4/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/mul_6Є
lstm/lstm_cell_4/mul_7Mullstm/zeros:output:0$lstm/lstm_cell_4/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/mul_7Ќ
lstm/lstm_cell_4/ReadVariableOpReadVariableOp(lstm_lstm_cell_4_readvariableop_resource*
_output_shapes
:	d*
dtype02!
lstm/lstm_cell_4/ReadVariableOp
$lstm/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm/lstm_cell_4/strided_slice/stackЁ
&lstm/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   2(
&lstm/lstm_cell_4/strided_slice/stack_1Ё
&lstm/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&lstm/lstm_cell_4/strided_slice/stack_2т
lstm/lstm_cell_4/strided_sliceStridedSlice'lstm/lstm_cell_4/ReadVariableOp:value:0-lstm/lstm_cell_4/strided_slice/stack:output:0/lstm/lstm_cell_4/strided_slice/stack_1:output:0/lstm/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2 
lstm/lstm_cell_4/strided_sliceЗ
lstm/lstm_cell_4/MatMul_4MatMullstm/lstm_cell_4/mul_4:z:0'lstm/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/MatMul_4Џ
lstm/lstm_cell_4/addAddV2!lstm/lstm_cell_4/BiasAdd:output:0#lstm/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/add
lstm/lstm_cell_4/SigmoidSigmoidlstm/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/SigmoidА
!lstm/lstm_cell_4/ReadVariableOp_1ReadVariableOp(lstm_lstm_cell_4_readvariableop_resource*
_output_shapes
:	d*
dtype02#
!lstm/lstm_cell_4/ReadVariableOp_1Ё
&lstm/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   2(
&lstm/lstm_cell_4/strided_slice_1/stackЅ
(lstm/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Ш   2*
(lstm/lstm_cell_4/strided_slice_1/stack_1Ѕ
(lstm/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(lstm/lstm_cell_4/strided_slice_1/stack_2ю
 lstm/lstm_cell_4/strided_slice_1StridedSlice)lstm/lstm_cell_4/ReadVariableOp_1:value:0/lstm/lstm_cell_4/strided_slice_1/stack:output:01lstm/lstm_cell_4/strided_slice_1/stack_1:output:01lstm/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2"
 lstm/lstm_cell_4/strided_slice_1Й
lstm/lstm_cell_4/MatMul_5MatMullstm/lstm_cell_4/mul_5:z:0)lstm/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/MatMul_5Е
lstm/lstm_cell_4/add_1AddV2#lstm/lstm_cell_4/BiasAdd_1:output:0#lstm/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/add_1
lstm/lstm_cell_4/Sigmoid_1Sigmoidlstm/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/Sigmoid_1 
lstm/lstm_cell_4/mul_8Mullstm/lstm_cell_4/Sigmoid_1:y:0lstm/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/mul_8А
!lstm/lstm_cell_4/ReadVariableOp_2ReadVariableOp(lstm_lstm_cell_4_readvariableop_resource*
_output_shapes
:	d*
dtype02#
!lstm/lstm_cell_4/ReadVariableOp_2Ё
&lstm/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    Ш   2(
&lstm/lstm_cell_4/strided_slice_2/stackЅ
(lstm/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  2*
(lstm/lstm_cell_4/strided_slice_2/stack_1Ѕ
(lstm/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(lstm/lstm_cell_4/strided_slice_2/stack_2ю
 lstm/lstm_cell_4/strided_slice_2StridedSlice)lstm/lstm_cell_4/ReadVariableOp_2:value:0/lstm/lstm_cell_4/strided_slice_2/stack:output:01lstm/lstm_cell_4/strided_slice_2/stack_1:output:01lstm/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2"
 lstm/lstm_cell_4/strided_slice_2Й
lstm/lstm_cell_4/MatMul_6MatMullstm/lstm_cell_4/mul_6:z:0)lstm/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/MatMul_6Е
lstm/lstm_cell_4/add_2AddV2#lstm/lstm_cell_4/BiasAdd_2:output:0#lstm/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/add_2
lstm/lstm_cell_4/TanhTanhlstm/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/TanhЂ
lstm/lstm_cell_4/mul_9Mullstm/lstm_cell_4/Sigmoid:y:0lstm/lstm_cell_4/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/mul_9Ѓ
lstm/lstm_cell_4/add_3AddV2lstm/lstm_cell_4/mul_8:z:0lstm/lstm_cell_4/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/add_3А
!lstm/lstm_cell_4/ReadVariableOp_3ReadVariableOp(lstm_lstm_cell_4_readvariableop_resource*
_output_shapes
:	d*
dtype02#
!lstm/lstm_cell_4/ReadVariableOp_3Ё
&lstm/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  2(
&lstm/lstm_cell_4/strided_slice_3/stackЅ
(lstm/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(lstm/lstm_cell_4/strided_slice_3/stack_1Ѕ
(lstm/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(lstm/lstm_cell_4/strided_slice_3/stack_2ю
 lstm/lstm_cell_4/strided_slice_3StridedSlice)lstm/lstm_cell_4/ReadVariableOp_3:value:0/lstm/lstm_cell_4/strided_slice_3/stack:output:01lstm/lstm_cell_4/strided_slice_3/stack_1:output:01lstm/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2"
 lstm/lstm_cell_4/strided_slice_3Й
lstm/lstm_cell_4/MatMul_7MatMullstm/lstm_cell_4/mul_7:z:0)lstm/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/MatMul_7Е
lstm/lstm_cell_4/add_4AddV2#lstm/lstm_cell_4/BiasAdd_3:output:0#lstm/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/add_4
lstm/lstm_cell_4/Sigmoid_2Sigmoidlstm/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/Sigmoid_2
lstm/lstm_cell_4/Tanh_1Tanhlstm/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/Tanh_1Ј
lstm/lstm_cell_4/mul_10Mullstm/lstm_cell_4/Sigmoid_2:y:0lstm/lstm_cell_4/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/mul_10
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   2$
"lstm/TensorArrayV2_1/element_shapeЬ
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2_1X
	lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
	lstm/time
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
lstm/while/maximum_iterationst
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm/while/loop_counterЊ

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0.lstm_lstm_cell_4_split_readvariableop_resource0lstm_lstm_cell_4_split_1_readvariableop_resource(lstm_lstm_cell_4_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *%
_read_only_resource_inputs
	
*!
bodyR
lstm_while_body_33057*!
condR
lstm_while_cond_33056*K
output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *
parallel_iterations 2

lstm/whileП
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   27
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeќ
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:(џџџџџџџџџd*
element_dtype02)
'lstm/TensorArrayV2Stack/TensorListStack
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
lstm/strided_slice_3/stack
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_3/stack_1
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_3/stack_2И
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_mask2
lstm/strided_slice_3
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose_1/permЙ
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ(d2
lstm/transpose_1p
lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/runtimeБ
!dense_final/MatMul/ReadVariableOpReadVariableOp*dense_final_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02#
!dense_final/MatMul/ReadVariableOpЎ
dense_final/MatMulMatMullstm/strided_slice_3:output:0)dense_final/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_final/MatMulА
"dense_final/BiasAdd/ReadVariableOpReadVariableOp+dense_final_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"dense_final/BiasAdd/ReadVariableOpБ
dense_final/BiasAddBiasAdddense_final/MatMul:product:0*dense_final/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_final/BiasAdd
dense_final/SoftmaxSoftmaxdense_final/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_final/Softmax~
IdentityIdentitydense_final/Softmax:softmax:0^lstm/while*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ(::::::2

lstm/while
lstm/while:O K
'
_output_shapes
:џџџџџџџџџ(
 
_user_specified_nameinputs
Ю
Н
#__inference_signature_wrapper_32868
embedding_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallembedding_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_312282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ(::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:џџџџџџџџџ(
)
_user_specified_nameembedding_input
п
Л
*__inference_sequential_layer_call_fn_33566

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityЂStatefulPartitionedCallЉ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_328262
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ(::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ(
 
_user_specified_nameinputs
т
l
P__inference_temporal_mean_pooling_layer_call_and_return_conditional_losses_34911

inputs
identityZ
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџd:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
е
Р
E__inference_sequential_layer_call_and_return_conditional_losses_32826

inputs
embedding_32809

lstm_32812

lstm_32814

lstm_32816
dense_final_32820
dense_final_32822
identityЂ#dense_final/StatefulPartitionedCallЂ!embedding/StatefulPartitionedCallЂlstm/StatefulPartitionedCall
!embedding/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_32809*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ(Ќ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_320182#
!embedding/StatefulPartitionedCallЏ
lstm/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0
lstm_32812
lstm_32814
lstm_32816*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_326672
lstm/StatefulPartitionedCall
%temporal_mean_pooling/PartitionedCallPartitionedCall%lstm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_temporal_mean_pooling_layer_call_and_return_conditional_losses_327052'
%temporal_mean_pooling/PartitionedCallШ
#dense_final/StatefulPartitionedCallStatefulPartitionedCall.temporal_mean_pooling/PartitionedCall:output:0dense_final_32820dense_final_32822*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_final_layer_call_and_return_conditional_losses_327292%
#dense_final/StatefulPartitionedCallщ
IdentityIdentity,dense_final/StatefulPartitionedCall:output:0$^dense_final/StatefulPartitionedCall"^embedding/StatefulPartitionedCall^lstm/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ(::::::2J
#dense_final/StatefulPartitionedCall#dense_final/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ(
 
_user_specified_nameinputs
ж
п
while_body_34745
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_05
1while_lstm_cell_4_split_readvariableop_resource_07
3while_lstm_cell_4_split_1_readvariableop_resource_0/
+while_lstm_cell_4_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor3
/while_lstm_cell_4_split_readvariableop_resource5
1while_lstm_cell_4_split_1_readvariableop_resource-
)while_lstm_cell_4_readvariableop_resourceУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeд
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџЌ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemІ
!while/lstm_cell_4/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/ones_like/Shape
!while/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!while/lstm_cell_4/ones_like/ConstЭ
while/lstm_cell_4/ones_likeFill*while/lstm_cell_4/ones_like/Shape:output:0*while/lstm_cell_4/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_4/ones_like
#while/lstm_cell_4/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2%
#while/lstm_cell_4/ones_like_1/Shape
#while/lstm_cell_4/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2%
#while/lstm_cell_4/ones_like_1/Constд
while/lstm_cell_4/ones_like_1Fill,while/lstm_cell_4/ones_like_1/Shape:output:0,while/lstm_cell_4/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/ones_like_1Р
while/lstm_cell_4/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_4/mulФ
while/lstm_cell_4/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_4/mul_1Ф
while/lstm_cell_4/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_4/mul_2Ф
while/lstm_cell_4/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_4/mul_3t
while/lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_4/Const
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_4/split/split_dimФ
&while/lstm_cell_4/split/ReadVariableOpReadVariableOp1while_lstm_cell_4_split_readvariableop_resource_0* 
_output_shapes
:
Ќ*
dtype02(
&while/lstm_cell_4/split/ReadVariableOpѓ
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0.while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	Ќd:	Ќd:	Ќd:	Ќd*
	num_split2
while/lstm_cell_4/split­
while/lstm_cell_4/MatMulMatMulwhile/lstm_cell_4/mul:z:0 while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMulГ
while/lstm_cell_4/MatMul_1MatMulwhile/lstm_cell_4/mul_1:z:0 while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMul_1Г
while/lstm_cell_4/MatMul_2MatMulwhile/lstm_cell_4/mul_2:z:0 while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMul_2Г
while/lstm_cell_4/MatMul_3MatMulwhile/lstm_cell_4/mul_3:z:0 while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMul_3x
while/lstm_cell_4/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_4/Const_1
#while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_4/split_1/split_dimХ
(while/lstm_cell_4/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02*
(while/lstm_cell_4/split_1/ReadVariableOpч
while/lstm_cell_4/split_1Split,while/lstm_cell_4/split_1/split_dim:output:00while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:d:d:d:d*
	num_split2
while/lstm_cell_4/split_1Л
while/lstm_cell_4/BiasAddBiasAdd"while/lstm_cell_4/MatMul:product:0"while/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/BiasAddС
while/lstm_cell_4/BiasAdd_1BiasAdd$while/lstm_cell_4/MatMul_1:product:0"while/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/BiasAdd_1С
while/lstm_cell_4/BiasAdd_2BiasAdd$while/lstm_cell_4/MatMul_2:product:0"while/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/BiasAdd_2С
while/lstm_cell_4/BiasAdd_3BiasAdd$while/lstm_cell_4/MatMul_3:product:0"while/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/BiasAdd_3Ј
while/lstm_cell_4/mul_4Mulwhile_placeholder_2&while/lstm_cell_4/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/mul_4Ј
while/lstm_cell_4/mul_5Mulwhile_placeholder_2&while/lstm_cell_4/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/mul_5Ј
while/lstm_cell_4/mul_6Mulwhile_placeholder_2&while/lstm_cell_4/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/mul_6Ј
while/lstm_cell_4/mul_7Mulwhile_placeholder_2&while/lstm_cell_4/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/mul_7Б
 while/lstm_cell_4/ReadVariableOpReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	d*
dtype02"
 while/lstm_cell_4/ReadVariableOp
%while/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell_4/strided_slice/stackЃ
'while/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   2)
'while/lstm_cell_4/strided_slice/stack_1Ѓ
'while/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell_4/strided_slice/stack_2ш
while/lstm_cell_4/strided_sliceStridedSlice(while/lstm_cell_4/ReadVariableOp:value:0.while/lstm_cell_4/strided_slice/stack:output:00while/lstm_cell_4/strided_slice/stack_1:output:00while/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2!
while/lstm_cell_4/strided_sliceЛ
while/lstm_cell_4/MatMul_4MatMulwhile/lstm_cell_4/mul_4:z:0(while/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMul_4Г
while/lstm_cell_4/addAddV2"while/lstm_cell_4/BiasAdd:output:0$while/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/add
while/lstm_cell_4/SigmoidSigmoidwhile/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/SigmoidЕ
"while/lstm_cell_4/ReadVariableOp_1ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	d*
dtype02$
"while/lstm_cell_4/ReadVariableOp_1Ѓ
'while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   2)
'while/lstm_cell_4/strided_slice_1/stackЇ
)while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Ш   2+
)while/lstm_cell_4/strided_slice_1/stack_1Ї
)while/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_4/strided_slice_1/stack_2є
!while/lstm_cell_4/strided_slice_1StridedSlice*while/lstm_cell_4/ReadVariableOp_1:value:00while/lstm_cell_4/strided_slice_1/stack:output:02while/lstm_cell_4/strided_slice_1/stack_1:output:02while/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_1Н
while/lstm_cell_4/MatMul_5MatMulwhile/lstm_cell_4/mul_5:z:0*while/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMul_5Й
while/lstm_cell_4/add_1AddV2$while/lstm_cell_4/BiasAdd_1:output:0$while/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/add_1
while/lstm_cell_4/Sigmoid_1Sigmoidwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/Sigmoid_1Ё
while/lstm_cell_4/mul_8Mulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/mul_8Е
"while/lstm_cell_4/ReadVariableOp_2ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	d*
dtype02$
"while/lstm_cell_4/ReadVariableOp_2Ѓ
'while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    Ш   2)
'while/lstm_cell_4/strided_slice_2/stackЇ
)while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  2+
)while/lstm_cell_4/strided_slice_2/stack_1Ї
)while/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_4/strided_slice_2/stack_2є
!while/lstm_cell_4/strided_slice_2StridedSlice*while/lstm_cell_4/ReadVariableOp_2:value:00while/lstm_cell_4/strided_slice_2/stack:output:02while/lstm_cell_4/strided_slice_2/stack_1:output:02while/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_2Н
while/lstm_cell_4/MatMul_6MatMulwhile/lstm_cell_4/mul_6:z:0*while/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMul_6Й
while/lstm_cell_4/add_2AddV2$while/lstm_cell_4/BiasAdd_2:output:0$while/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/add_2
while/lstm_cell_4/TanhTanhwhile/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/TanhІ
while/lstm_cell_4/mul_9Mulwhile/lstm_cell_4/Sigmoid:y:0while/lstm_cell_4/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/mul_9Ї
while/lstm_cell_4/add_3AddV2while/lstm_cell_4/mul_8:z:0while/lstm_cell_4/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/add_3Е
"while/lstm_cell_4/ReadVariableOp_3ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	d*
dtype02$
"while/lstm_cell_4/ReadVariableOp_3Ѓ
'while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  2)
'while/lstm_cell_4/strided_slice_3/stackЇ
)while/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_4/strided_slice_3/stack_1Ї
)while/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_4/strided_slice_3/stack_2є
!while/lstm_cell_4/strided_slice_3StridedSlice*while/lstm_cell_4/ReadVariableOp_3:value:00while/lstm_cell_4/strided_slice_3/stack:output:02while/lstm_cell_4/strided_slice_3/stack_1:output:02while/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_3Н
while/lstm_cell_4/MatMul_7MatMulwhile/lstm_cell_4/mul_7:z:0*while/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMul_7Й
while/lstm_cell_4/add_4AddV2$while/lstm_cell_4/BiasAdd_3:output:0$while/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/add_4
while/lstm_cell_4/Sigmoid_2Sigmoidwhile/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/Sigmoid_2
while/lstm_cell_4/Tanh_1Tanhwhile/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/Tanh_1Ќ
while/lstm_cell_4/mul_10Mulwhile/lstm_cell_4/Sigmoid_2:y:0while/lstm_cell_4/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/mul_10р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_4/mul_10:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_4/mul_10:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_4_readvariableop_resource+while_lstm_cell_4_readvariableop_resource_0"h
1while_lstm_cell_4_split_1_readvariableop_resource3while_lstm_cell_4_split_1_readvariableop_resource_0"d
/while_lstm_cell_4_split_readvariableop_resource1while_lstm_cell_4_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџd:џџџџџџџџџd: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
: 
нH

F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_31500

inputs

states
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2X
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like/Const
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
	ones_like\
ones_like_1/ShapeShapestates*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like_1/Const
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
ones_like_1`
mulMulinputsones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
muld
mul_1Mulinputsones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
mul_1d
mul_2Mulinputsones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
mul_2d
mul_3Mulinputsones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
mul_3P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
Ќ*
dtype02
split/ReadVariableOpЋ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	Ќd:	Ќd:	Ќd:	Ќd*
	num_split2
splite
MatMulMatMulmul:z:0split:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
MatMulk
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd2

MatMul_1k
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd2

MatMul_2k
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*'
_output_shapes
:џџџџџџџџџd2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype02
split_1/ReadVariableOp
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:d:d:d:d*
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџd2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџd2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџd2
	BiasAdd_3e
mul_4Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
mul_4e
mul_5Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
mul_5e
mul_6Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
mul_6e
mul_7Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
mul_7y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	d*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2ќ
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2
strided_slices
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2	
Sigmoid}
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	d*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Ш   2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2
strided_slice_1u
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2

MatMul_5q
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
	Sigmoid_1`
mul_8MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџd2
mul_8}
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	d*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    Ш   2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2
strided_slice_2u
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2

MatMul_6q
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
Tanh^
mul_9MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
mul_9_
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
add_3}
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	d*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  2
strided_slice_3/stack
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2
strided_slice_3u
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2

MatMul_7q
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
	Sigmoid_2U
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
Tanh_1d
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
mul_10^
IdentityIdentity
mul_10:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2

Identityb

Identity_1Identity
mul_10:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity_1a

Identity_2Identity	add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:џџџџџџџџџЌ:џџџџџџџџџd:џџџџџџџџџd::::P L
(
_output_shapes
:џџџџџџџџџЌ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_namestates:OK
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_namestates
Н	
Ђ
lstm_while_cond_33388&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3(
$lstm_while_less_lstm_strided_slice_1=
9lstm_while_lstm_while_cond_33388___redundant_placeholder0=
9lstm_while_lstm_while_cond_33388___redundant_placeholder1=
9lstm_while_lstm_while_cond_33388___redundant_placeholder2=
9lstm_while_lstm_while_cond_33388___redundant_placeholder3
lstm_while_identity

lstm/while/LessLesslstm_while_placeholder$lstm_while_less_lstm_strided_slice_1*
T0*
_output_shapes
: 2
lstm/while/Lessl
lstm/while/IdentityIdentitylstm/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm/while/Identity"3
lstm_while_identitylstm/while/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџd:џџџџџџџџџd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
:
К
я
?__inference_lstm_layer_call_and_return_conditional_losses_34881

inputs-
)lstm_cell_4_split_readvariableop_resource/
+lstm_cell_4_split_1_readvariableop_resource'
#lstm_cell_4_readvariableop_resource
identityЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:(џџџџџџџџџЌ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2§
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
shrink_axis_mask2
strided_slice_2
lstm_cell_4/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell_4/ones_like/Shape
lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_4/ones_like/ConstЕ
lstm_cell_4/ones_likeFill$lstm_cell_4/ones_like/Shape:output:0$lstm_cell_4/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/ones_like|
lstm_cell_4/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_4/ones_like_1/Shape
lstm_cell_4/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_4/ones_like_1/ConstМ
lstm_cell_4/ones_like_1Fill&lstm_cell_4/ones_like_1/Shape:output:0&lstm_cell_4/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/ones_like_1
lstm_cell_4/mulMulstrided_slice_2:output:0lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/mul
lstm_cell_4/mul_1Mulstrided_slice_2:output:0lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/mul_1
lstm_cell_4/mul_2Mulstrided_slice_2:output:0lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/mul_2
lstm_cell_4/mul_3Mulstrided_slice_2:output:0lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/mul_3h
lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/Const|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dimА
 lstm_cell_4/split/ReadVariableOpReadVariableOp)lstm_cell_4_split_readvariableop_resource* 
_output_shapes
:
Ќ*
dtype02"
 lstm_cell_4/split/ReadVariableOpл
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0(lstm_cell_4/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	Ќd:	Ќd:	Ќd:	Ќd*
	num_split2
lstm_cell_4/split
lstm_cell_4/MatMulMatMullstm_cell_4/mul:z:0lstm_cell_4/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul
lstm_cell_4/MatMul_1MatMullstm_cell_4/mul_1:z:0lstm_cell_4/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul_1
lstm_cell_4/MatMul_2MatMullstm_cell_4/mul_2:z:0lstm_cell_4/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul_2
lstm_cell_4/MatMul_3MatMullstm_cell_4/mul_3:z:0lstm_cell_4/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul_3l
lstm_cell_4/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/Const_1
lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_4/split_1/split_dimБ
"lstm_cell_4/split_1/ReadVariableOpReadVariableOp+lstm_cell_4_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02$
"lstm_cell_4/split_1/ReadVariableOpЯ
lstm_cell_4/split_1Split&lstm_cell_4/split_1/split_dim:output:0*lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:d:d:d:d*
	num_split2
lstm_cell_4/split_1Ѓ
lstm_cell_4/BiasAddBiasAddlstm_cell_4/MatMul:product:0lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/BiasAddЉ
lstm_cell_4/BiasAdd_1BiasAddlstm_cell_4/MatMul_1:product:0lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/BiasAdd_1Љ
lstm_cell_4/BiasAdd_2BiasAddlstm_cell_4/MatMul_2:product:0lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/BiasAdd_2Љ
lstm_cell_4/BiasAdd_3BiasAddlstm_cell_4/MatMul_3:product:0lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/BiasAdd_3
lstm_cell_4/mul_4Mulzeros:output:0 lstm_cell_4/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/mul_4
lstm_cell_4/mul_5Mulzeros:output:0 lstm_cell_4/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/mul_5
lstm_cell_4/mul_6Mulzeros:output:0 lstm_cell_4/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/mul_6
lstm_cell_4/mul_7Mulzeros:output:0 lstm_cell_4/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/mul_7
lstm_cell_4/ReadVariableOpReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	d*
dtype02
lstm_cell_4/ReadVariableOp
lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell_4/strided_slice/stack
!lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   2#
!lstm_cell_4/strided_slice/stack_1
!lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell_4/strided_slice/stack_2Ф
lstm_cell_4/strided_sliceStridedSlice"lstm_cell_4/ReadVariableOp:value:0(lstm_cell_4/strided_slice/stack:output:0*lstm_cell_4/strided_slice/stack_1:output:0*lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2
lstm_cell_4/strided_sliceЃ
lstm_cell_4/MatMul_4MatMullstm_cell_4/mul_4:z:0"lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul_4
lstm_cell_4/addAddV2lstm_cell_4/BiasAdd:output:0lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/add|
lstm_cell_4/SigmoidSigmoidlstm_cell_4/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/SigmoidЁ
lstm_cell_4/ReadVariableOp_1ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	d*
dtype02
lstm_cell_4/ReadVariableOp_1
!lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   2#
!lstm_cell_4/strided_slice_1/stack
#lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Ш   2%
#lstm_cell_4/strided_slice_1/stack_1
#lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_4/strided_slice_1/stack_2а
lstm_cell_4/strided_slice_1StridedSlice$lstm_cell_4/ReadVariableOp_1:value:0*lstm_cell_4/strided_slice_1/stack:output:0,lstm_cell_4/strided_slice_1/stack_1:output:0,lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2
lstm_cell_4/strided_slice_1Ѕ
lstm_cell_4/MatMul_5MatMullstm_cell_4/mul_5:z:0$lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul_5Ё
lstm_cell_4/add_1AddV2lstm_cell_4/BiasAdd_1:output:0lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/add_1
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/Sigmoid_1
lstm_cell_4/mul_8Mullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/mul_8Ё
lstm_cell_4/ReadVariableOp_2ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	d*
dtype02
lstm_cell_4/ReadVariableOp_2
!lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    Ш   2#
!lstm_cell_4/strided_slice_2/stack
#lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  2%
#lstm_cell_4/strided_slice_2/stack_1
#lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_4/strided_slice_2/stack_2а
lstm_cell_4/strided_slice_2StridedSlice$lstm_cell_4/ReadVariableOp_2:value:0*lstm_cell_4/strided_slice_2/stack:output:0,lstm_cell_4/strided_slice_2/stack_1:output:0,lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2
lstm_cell_4/strided_slice_2Ѕ
lstm_cell_4/MatMul_6MatMullstm_cell_4/mul_6:z:0$lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul_6Ё
lstm_cell_4/add_2AddV2lstm_cell_4/BiasAdd_2:output:0lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/add_2u
lstm_cell_4/TanhTanhlstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/Tanh
lstm_cell_4/mul_9Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/mul_9
lstm_cell_4/add_3AddV2lstm_cell_4/mul_8:z:0lstm_cell_4/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/add_3Ё
lstm_cell_4/ReadVariableOp_3ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	d*
dtype02
lstm_cell_4/ReadVariableOp_3
!lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  2#
!lstm_cell_4/strided_slice_3/stack
#lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_4/strided_slice_3/stack_1
#lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_4/strided_slice_3/stack_2а
lstm_cell_4/strided_slice_3StridedSlice$lstm_cell_4/ReadVariableOp_3:value:0*lstm_cell_4/strided_slice_3/stack:output:0,lstm_cell_4/strided_slice_3/stack_1:output:0,lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2
lstm_cell_4/strided_slice_3Ѕ
lstm_cell_4/MatMul_7MatMullstm_cell_4/mul_7:z:0$lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul_7Ё
lstm_cell_4/add_4AddV2lstm_cell_4/BiasAdd_3:output:0lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/add_4
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Tanh_1Tanhlstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/Tanh_1
lstm_cell_4/mul_10Mullstm_cell_4/Sigmoid_2:y:0lstm_cell_4/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/mul_10
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterп
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_4_split_readvariableop_resource+lstm_cell_4_split_1_readvariableop_resource#lstm_cell_4_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_34745*
condR
while_cond_34744*K
output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:(џџџџџџџџџd*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЅ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ(d2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:џџџџџџџџџ(Ќ:::2
whilewhile:T P
,
_output_shapes
:џџџџџџџџџ(Ќ
 
_user_specified_nameinputs
т
l
P__inference_temporal_mean_pooling_layer_call_and_return_conditional_losses_34907

inputs
identityZ
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџd:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
Мё
я
?__inference_lstm_layer_call_and_return_conditional_losses_34626

inputs-
)lstm_cell_4_split_readvariableop_resource/
+lstm_cell_4_split_1_readvariableop_resource'
#lstm_cell_4_readvariableop_resource
identityЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:(џџџџџџџџџЌ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2§
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
shrink_axis_mask2
strided_slice_2
lstm_cell_4/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell_4/ones_like/Shape
lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_4/ones_like/ConstЕ
lstm_cell_4/ones_likeFill$lstm_cell_4/ones_like/Shape:output:0$lstm_cell_4/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/ones_like{
lstm_cell_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2
lstm_cell_4/dropout/ConstА
lstm_cell_4/dropout/MulMullstm_cell_4/ones_like:output:0"lstm_cell_4/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/dropout/Mul
lstm_cell_4/dropout/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout/Shapeј
0lstm_cell_4/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_4/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*
seedБџх)*
seed2ТнШ22
0lstm_cell_4/dropout/random_uniform/RandomUniform
"lstm_cell_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2$
"lstm_cell_4/dropout/GreaterEqual/yя
 lstm_cell_4/dropout/GreaterEqualGreaterEqual9lstm_cell_4/dropout/random_uniform/RandomUniform:output:0+lstm_cell_4/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2"
 lstm_cell_4/dropout/GreaterEqualЄ
lstm_cell_4/dropout/CastCast$lstm_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/dropout/CastЋ
lstm_cell_4/dropout/Mul_1Mullstm_cell_4/dropout/Mul:z:0lstm_cell_4/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/dropout/Mul_1
lstm_cell_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2
lstm_cell_4/dropout_1/ConstЖ
lstm_cell_4/dropout_1/MulMullstm_cell_4/ones_like:output:0$lstm_cell_4/dropout_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/dropout_1/Mul
lstm_cell_4/dropout_1/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout_1/Shape§
2lstm_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_1/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*
seedБџх)*
seed2Њw24
2lstm_cell_4/dropout_1/random_uniform/RandomUniform
$lstm_cell_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2&
$lstm_cell_4/dropout_1/GreaterEqual/yї
"lstm_cell_4/dropout_1/GreaterEqualGreaterEqual;lstm_cell_4/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2$
"lstm_cell_4/dropout_1/GreaterEqualЊ
lstm_cell_4/dropout_1/CastCast&lstm_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/dropout_1/CastГ
lstm_cell_4/dropout_1/Mul_1Mullstm_cell_4/dropout_1/Mul:z:0lstm_cell_4/dropout_1/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/dropout_1/Mul_1
lstm_cell_4/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2
lstm_cell_4/dropout_2/ConstЖ
lstm_cell_4/dropout_2/MulMullstm_cell_4/ones_like:output:0$lstm_cell_4/dropout_2/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/dropout_2/Mul
lstm_cell_4/dropout_2/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout_2/Shapeў
2lstm_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_2/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*
seedБџх)*
seed2ѓБ24
2lstm_cell_4/dropout_2/random_uniform/RandomUniform
$lstm_cell_4/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2&
$lstm_cell_4/dropout_2/GreaterEqual/yї
"lstm_cell_4/dropout_2/GreaterEqualGreaterEqual;lstm_cell_4/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2$
"lstm_cell_4/dropout_2/GreaterEqualЊ
lstm_cell_4/dropout_2/CastCast&lstm_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/dropout_2/CastГ
lstm_cell_4/dropout_2/Mul_1Mullstm_cell_4/dropout_2/Mul:z:0lstm_cell_4/dropout_2/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/dropout_2/Mul_1
lstm_cell_4/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2
lstm_cell_4/dropout_3/ConstЖ
lstm_cell_4/dropout_3/MulMullstm_cell_4/ones_like:output:0$lstm_cell_4/dropout_3/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/dropout_3/Mul
lstm_cell_4/dropout_3/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout_3/Shapeў
2lstm_cell_4/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_3/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*
seedБџх)*
seed2Зп24
2lstm_cell_4/dropout_3/random_uniform/RandomUniform
$lstm_cell_4/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2&
$lstm_cell_4/dropout_3/GreaterEqual/yї
"lstm_cell_4/dropout_3/GreaterEqualGreaterEqual;lstm_cell_4/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2$
"lstm_cell_4/dropout_3/GreaterEqualЊ
lstm_cell_4/dropout_3/CastCast&lstm_cell_4/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/dropout_3/CastГ
lstm_cell_4/dropout_3/Mul_1Mullstm_cell_4/dropout_3/Mul:z:0lstm_cell_4/dropout_3/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/dropout_3/Mul_1|
lstm_cell_4/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_4/ones_like_1/Shape
lstm_cell_4/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_4/ones_like_1/ConstМ
lstm_cell_4/ones_like_1Fill&lstm_cell_4/ones_like_1/Shape:output:0&lstm_cell_4/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/ones_like_1
lstm_cell_4/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2
lstm_cell_4/dropout_4/ConstЗ
lstm_cell_4/dropout_4/MulMul lstm_cell_4/ones_like_1:output:0$lstm_cell_4/dropout_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/dropout_4/Mul
lstm_cell_4/dropout_4/ShapeShape lstm_cell_4/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout_4/Shape§
2lstm_cell_4/dropout_4/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџd*
dtype0*
seedБџх)*
seed2МЄ24
2lstm_cell_4/dropout_4/random_uniform/RandomUniform
$lstm_cell_4/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2&
$lstm_cell_4/dropout_4/GreaterEqual/yі
"lstm_cell_4/dropout_4/GreaterEqualGreaterEqual;lstm_cell_4/dropout_4/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2$
"lstm_cell_4/dropout_4/GreaterEqualЉ
lstm_cell_4/dropout_4/CastCast&lstm_cell_4/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/dropout_4/CastВ
lstm_cell_4/dropout_4/Mul_1Mullstm_cell_4/dropout_4/Mul:z:0lstm_cell_4/dropout_4/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/dropout_4/Mul_1
lstm_cell_4/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2
lstm_cell_4/dropout_5/ConstЗ
lstm_cell_4/dropout_5/MulMul lstm_cell_4/ones_like_1:output:0$lstm_cell_4/dropout_5/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/dropout_5/Mul
lstm_cell_4/dropout_5/ShapeShape lstm_cell_4/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout_5/Shape§
2lstm_cell_4/dropout_5/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџd*
dtype0*
seedБџх)*
seed2Ч 24
2lstm_cell_4/dropout_5/random_uniform/RandomUniform
$lstm_cell_4/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2&
$lstm_cell_4/dropout_5/GreaterEqual/yі
"lstm_cell_4/dropout_5/GreaterEqualGreaterEqual;lstm_cell_4/dropout_5/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2$
"lstm_cell_4/dropout_5/GreaterEqualЉ
lstm_cell_4/dropout_5/CastCast&lstm_cell_4/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/dropout_5/CastВ
lstm_cell_4/dropout_5/Mul_1Mullstm_cell_4/dropout_5/Mul:z:0lstm_cell_4/dropout_5/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/dropout_5/Mul_1
lstm_cell_4/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2
lstm_cell_4/dropout_6/ConstЗ
lstm_cell_4/dropout_6/MulMul lstm_cell_4/ones_like_1:output:0$lstm_cell_4/dropout_6/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/dropout_6/Mul
lstm_cell_4/dropout_6/ShapeShape lstm_cell_4/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout_6/Shape§
2lstm_cell_4/dropout_6/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_6/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџd*
dtype0*
seedБџх)*
seed2ї24
2lstm_cell_4/dropout_6/random_uniform/RandomUniform
$lstm_cell_4/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2&
$lstm_cell_4/dropout_6/GreaterEqual/yі
"lstm_cell_4/dropout_6/GreaterEqualGreaterEqual;lstm_cell_4/dropout_6/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2$
"lstm_cell_4/dropout_6/GreaterEqualЉ
lstm_cell_4/dropout_6/CastCast&lstm_cell_4/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/dropout_6/CastВ
lstm_cell_4/dropout_6/Mul_1Mullstm_cell_4/dropout_6/Mul:z:0lstm_cell_4/dropout_6/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/dropout_6/Mul_1
lstm_cell_4/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2
lstm_cell_4/dropout_7/ConstЗ
lstm_cell_4/dropout_7/MulMul lstm_cell_4/ones_like_1:output:0$lstm_cell_4/dropout_7/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/dropout_7/Mul
lstm_cell_4/dropout_7/ShapeShape lstm_cell_4/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout_7/Shapeќ
2lstm_cell_4/dropout_7/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_7/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџd*
dtype0*
seedБџх)*
seed2Іяj24
2lstm_cell_4/dropout_7/random_uniform/RandomUniform
$lstm_cell_4/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2&
$lstm_cell_4/dropout_7/GreaterEqual/yі
"lstm_cell_4/dropout_7/GreaterEqualGreaterEqual;lstm_cell_4/dropout_7/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2$
"lstm_cell_4/dropout_7/GreaterEqualЉ
lstm_cell_4/dropout_7/CastCast&lstm_cell_4/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/dropout_7/CastВ
lstm_cell_4/dropout_7/Mul_1Mullstm_cell_4/dropout_7/Mul:z:0lstm_cell_4/dropout_7/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/dropout_7/Mul_1
lstm_cell_4/mulMulstrided_slice_2:output:0lstm_cell_4/dropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/mul
lstm_cell_4/mul_1Mulstrided_slice_2:output:0lstm_cell_4/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/mul_1
lstm_cell_4/mul_2Mulstrided_slice_2:output:0lstm_cell_4/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/mul_2
lstm_cell_4/mul_3Mulstrided_slice_2:output:0lstm_cell_4/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/mul_3h
lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/Const|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dimА
 lstm_cell_4/split/ReadVariableOpReadVariableOp)lstm_cell_4_split_readvariableop_resource* 
_output_shapes
:
Ќ*
dtype02"
 lstm_cell_4/split/ReadVariableOpл
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0(lstm_cell_4/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	Ќd:	Ќd:	Ќd:	Ќd*
	num_split2
lstm_cell_4/split
lstm_cell_4/MatMulMatMullstm_cell_4/mul:z:0lstm_cell_4/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul
lstm_cell_4/MatMul_1MatMullstm_cell_4/mul_1:z:0lstm_cell_4/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul_1
lstm_cell_4/MatMul_2MatMullstm_cell_4/mul_2:z:0lstm_cell_4/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul_2
lstm_cell_4/MatMul_3MatMullstm_cell_4/mul_3:z:0lstm_cell_4/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul_3l
lstm_cell_4/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/Const_1
lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_4/split_1/split_dimБ
"lstm_cell_4/split_1/ReadVariableOpReadVariableOp+lstm_cell_4_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02$
"lstm_cell_4/split_1/ReadVariableOpЯ
lstm_cell_4/split_1Split&lstm_cell_4/split_1/split_dim:output:0*lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:d:d:d:d*
	num_split2
lstm_cell_4/split_1Ѓ
lstm_cell_4/BiasAddBiasAddlstm_cell_4/MatMul:product:0lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/BiasAddЉ
lstm_cell_4/BiasAdd_1BiasAddlstm_cell_4/MatMul_1:product:0lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/BiasAdd_1Љ
lstm_cell_4/BiasAdd_2BiasAddlstm_cell_4/MatMul_2:product:0lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/BiasAdd_2Љ
lstm_cell_4/BiasAdd_3BiasAddlstm_cell_4/MatMul_3:product:0lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/BiasAdd_3
lstm_cell_4/mul_4Mulzeros:output:0lstm_cell_4/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/mul_4
lstm_cell_4/mul_5Mulzeros:output:0lstm_cell_4/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/mul_5
lstm_cell_4/mul_6Mulzeros:output:0lstm_cell_4/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/mul_6
lstm_cell_4/mul_7Mulzeros:output:0lstm_cell_4/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/mul_7
lstm_cell_4/ReadVariableOpReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	d*
dtype02
lstm_cell_4/ReadVariableOp
lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell_4/strided_slice/stack
!lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   2#
!lstm_cell_4/strided_slice/stack_1
!lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell_4/strided_slice/stack_2Ф
lstm_cell_4/strided_sliceStridedSlice"lstm_cell_4/ReadVariableOp:value:0(lstm_cell_4/strided_slice/stack:output:0*lstm_cell_4/strided_slice/stack_1:output:0*lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2
lstm_cell_4/strided_sliceЃ
lstm_cell_4/MatMul_4MatMullstm_cell_4/mul_4:z:0"lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul_4
lstm_cell_4/addAddV2lstm_cell_4/BiasAdd:output:0lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/add|
lstm_cell_4/SigmoidSigmoidlstm_cell_4/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/SigmoidЁ
lstm_cell_4/ReadVariableOp_1ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	d*
dtype02
lstm_cell_4/ReadVariableOp_1
!lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   2#
!lstm_cell_4/strided_slice_1/stack
#lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Ш   2%
#lstm_cell_4/strided_slice_1/stack_1
#lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_4/strided_slice_1/stack_2а
lstm_cell_4/strided_slice_1StridedSlice$lstm_cell_4/ReadVariableOp_1:value:0*lstm_cell_4/strided_slice_1/stack:output:0,lstm_cell_4/strided_slice_1/stack_1:output:0,lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2
lstm_cell_4/strided_slice_1Ѕ
lstm_cell_4/MatMul_5MatMullstm_cell_4/mul_5:z:0$lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul_5Ё
lstm_cell_4/add_1AddV2lstm_cell_4/BiasAdd_1:output:0lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/add_1
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/Sigmoid_1
lstm_cell_4/mul_8Mullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/mul_8Ё
lstm_cell_4/ReadVariableOp_2ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	d*
dtype02
lstm_cell_4/ReadVariableOp_2
!lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    Ш   2#
!lstm_cell_4/strided_slice_2/stack
#lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  2%
#lstm_cell_4/strided_slice_2/stack_1
#lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_4/strided_slice_2/stack_2а
lstm_cell_4/strided_slice_2StridedSlice$lstm_cell_4/ReadVariableOp_2:value:0*lstm_cell_4/strided_slice_2/stack:output:0,lstm_cell_4/strided_slice_2/stack_1:output:0,lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2
lstm_cell_4/strided_slice_2Ѕ
lstm_cell_4/MatMul_6MatMullstm_cell_4/mul_6:z:0$lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul_6Ё
lstm_cell_4/add_2AddV2lstm_cell_4/BiasAdd_2:output:0lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/add_2u
lstm_cell_4/TanhTanhlstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/Tanh
lstm_cell_4/mul_9Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/mul_9
lstm_cell_4/add_3AddV2lstm_cell_4/mul_8:z:0lstm_cell_4/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/add_3Ё
lstm_cell_4/ReadVariableOp_3ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	d*
dtype02
lstm_cell_4/ReadVariableOp_3
!lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  2#
!lstm_cell_4/strided_slice_3/stack
#lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_4/strided_slice_3/stack_1
#lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_4/strided_slice_3/stack_2а
lstm_cell_4/strided_slice_3StridedSlice$lstm_cell_4/ReadVariableOp_3:value:0*lstm_cell_4/strided_slice_3/stack:output:0,lstm_cell_4/strided_slice_3/stack_1:output:0,lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2
lstm_cell_4/strided_slice_3Ѕ
lstm_cell_4/MatMul_7MatMullstm_cell_4/mul_7:z:0$lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul_7Ё
lstm_cell_4/add_4AddV2lstm_cell_4/BiasAdd_3:output:0lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/add_4
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Tanh_1Tanhlstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/Tanh_1
lstm_cell_4/mul_10Mullstm_cell_4/Sigmoid_2:y:0lstm_cell_4/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/mul_10
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterп
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_4_split_readvariableop_resource+lstm_cell_4_split_1_readvariableop_resource#lstm_cell_4_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_34426*
condR
while_cond_34425*K
output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:(џџџџџџџџџd*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЅ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ(d2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:џџџџџџџџџ(Ќ:::2
whilewhile:T P
,
_output_shapes
:џџџџџџџџџ(Ќ
 
_user_specified_nameinputs
яё
ё
?__inference_lstm_layer_call_and_return_conditional_losses_33966
inputs_0-
)lstm_cell_4_split_readvariableop_resource/
+lstm_cell_4_split_1_readvariableop_resource'
#lstm_cell_4_readvariableop_resource
identityЂwhileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2§
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
shrink_axis_mask2
strided_slice_2
lstm_cell_4/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell_4/ones_like/Shape
lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_4/ones_like/ConstЕ
lstm_cell_4/ones_likeFill$lstm_cell_4/ones_like/Shape:output:0$lstm_cell_4/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/ones_like{
lstm_cell_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2
lstm_cell_4/dropout/ConstА
lstm_cell_4/dropout/MulMullstm_cell_4/ones_like:output:0"lstm_cell_4/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/dropout/Mul
lstm_cell_4/dropout/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout/Shapeї
0lstm_cell_4/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_4/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*
seedБџх)*
seed2тЗ22
0lstm_cell_4/dropout/random_uniform/RandomUniform
"lstm_cell_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2$
"lstm_cell_4/dropout/GreaterEqual/yя
 lstm_cell_4/dropout/GreaterEqualGreaterEqual9lstm_cell_4/dropout/random_uniform/RandomUniform:output:0+lstm_cell_4/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2"
 lstm_cell_4/dropout/GreaterEqualЄ
lstm_cell_4/dropout/CastCast$lstm_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/dropout/CastЋ
lstm_cell_4/dropout/Mul_1Mullstm_cell_4/dropout/Mul:z:0lstm_cell_4/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/dropout/Mul_1
lstm_cell_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2
lstm_cell_4/dropout_1/ConstЖ
lstm_cell_4/dropout_1/MulMullstm_cell_4/ones_like:output:0$lstm_cell_4/dropout_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/dropout_1/Mul
lstm_cell_4/dropout_1/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout_1/Shape§
2lstm_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_1/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*
seedБџх)*
seed2ЯЖ624
2lstm_cell_4/dropout_1/random_uniform/RandomUniform
$lstm_cell_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2&
$lstm_cell_4/dropout_1/GreaterEqual/yї
"lstm_cell_4/dropout_1/GreaterEqualGreaterEqual;lstm_cell_4/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2$
"lstm_cell_4/dropout_1/GreaterEqualЊ
lstm_cell_4/dropout_1/CastCast&lstm_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/dropout_1/CastГ
lstm_cell_4/dropout_1/Mul_1Mullstm_cell_4/dropout_1/Mul:z:0lstm_cell_4/dropout_1/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/dropout_1/Mul_1
lstm_cell_4/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2
lstm_cell_4/dropout_2/ConstЖ
lstm_cell_4/dropout_2/MulMullstm_cell_4/ones_like:output:0$lstm_cell_4/dropout_2/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/dropout_2/Mul
lstm_cell_4/dropout_2/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout_2/Shape§
2lstm_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_2/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*
seedБџх)*
seed2єd24
2lstm_cell_4/dropout_2/random_uniform/RandomUniform
$lstm_cell_4/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2&
$lstm_cell_4/dropout_2/GreaterEqual/yї
"lstm_cell_4/dropout_2/GreaterEqualGreaterEqual;lstm_cell_4/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2$
"lstm_cell_4/dropout_2/GreaterEqualЊ
lstm_cell_4/dropout_2/CastCast&lstm_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/dropout_2/CastГ
lstm_cell_4/dropout_2/Mul_1Mullstm_cell_4/dropout_2/Mul:z:0lstm_cell_4/dropout_2/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/dropout_2/Mul_1
lstm_cell_4/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2
lstm_cell_4/dropout_3/ConstЖ
lstm_cell_4/dropout_3/MulMullstm_cell_4/ones_like:output:0$lstm_cell_4/dropout_3/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/dropout_3/Mul
lstm_cell_4/dropout_3/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout_3/Shapeў
2lstm_cell_4/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_3/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*
seedБџх)*
seed2ій24
2lstm_cell_4/dropout_3/random_uniform/RandomUniform
$lstm_cell_4/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2&
$lstm_cell_4/dropout_3/GreaterEqual/yї
"lstm_cell_4/dropout_3/GreaterEqualGreaterEqual;lstm_cell_4/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2$
"lstm_cell_4/dropout_3/GreaterEqualЊ
lstm_cell_4/dropout_3/CastCast&lstm_cell_4/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/dropout_3/CastГ
lstm_cell_4/dropout_3/Mul_1Mullstm_cell_4/dropout_3/Mul:z:0lstm_cell_4/dropout_3/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/dropout_3/Mul_1|
lstm_cell_4/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_4/ones_like_1/Shape
lstm_cell_4/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_4/ones_like_1/ConstМ
lstm_cell_4/ones_like_1Fill&lstm_cell_4/ones_like_1/Shape:output:0&lstm_cell_4/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/ones_like_1
lstm_cell_4/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2
lstm_cell_4/dropout_4/ConstЗ
lstm_cell_4/dropout_4/MulMul lstm_cell_4/ones_like_1:output:0$lstm_cell_4/dropout_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/dropout_4/Mul
lstm_cell_4/dropout_4/ShapeShape lstm_cell_4/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout_4/Shape§
2lstm_cell_4/dropout_4/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџd*
dtype0*
seedБџх)*
seed2юВЭ24
2lstm_cell_4/dropout_4/random_uniform/RandomUniform
$lstm_cell_4/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2&
$lstm_cell_4/dropout_4/GreaterEqual/yі
"lstm_cell_4/dropout_4/GreaterEqualGreaterEqual;lstm_cell_4/dropout_4/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2$
"lstm_cell_4/dropout_4/GreaterEqualЉ
lstm_cell_4/dropout_4/CastCast&lstm_cell_4/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/dropout_4/CastВ
lstm_cell_4/dropout_4/Mul_1Mullstm_cell_4/dropout_4/Mul:z:0lstm_cell_4/dropout_4/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/dropout_4/Mul_1
lstm_cell_4/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2
lstm_cell_4/dropout_5/ConstЗ
lstm_cell_4/dropout_5/MulMul lstm_cell_4/ones_like_1:output:0$lstm_cell_4/dropout_5/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/dropout_5/Mul
lstm_cell_4/dropout_5/ShapeShape lstm_cell_4/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout_5/Shapeќ
2lstm_cell_4/dropout_5/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџd*
dtype0*
seedБџх)*
seed2шд=24
2lstm_cell_4/dropout_5/random_uniform/RandomUniform
$lstm_cell_4/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2&
$lstm_cell_4/dropout_5/GreaterEqual/yі
"lstm_cell_4/dropout_5/GreaterEqualGreaterEqual;lstm_cell_4/dropout_5/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2$
"lstm_cell_4/dropout_5/GreaterEqualЉ
lstm_cell_4/dropout_5/CastCast&lstm_cell_4/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/dropout_5/CastВ
lstm_cell_4/dropout_5/Mul_1Mullstm_cell_4/dropout_5/Mul:z:0lstm_cell_4/dropout_5/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/dropout_5/Mul_1
lstm_cell_4/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2
lstm_cell_4/dropout_6/ConstЗ
lstm_cell_4/dropout_6/MulMul lstm_cell_4/ones_like_1:output:0$lstm_cell_4/dropout_6/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/dropout_6/Mul
lstm_cell_4/dropout_6/ShapeShape lstm_cell_4/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout_6/Shapeќ
2lstm_cell_4/dropout_6/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_6/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџd*
dtype0*
seedБџх)*
seed2ќ;24
2lstm_cell_4/dropout_6/random_uniform/RandomUniform
$lstm_cell_4/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2&
$lstm_cell_4/dropout_6/GreaterEqual/yі
"lstm_cell_4/dropout_6/GreaterEqualGreaterEqual;lstm_cell_4/dropout_6/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2$
"lstm_cell_4/dropout_6/GreaterEqualЉ
lstm_cell_4/dropout_6/CastCast&lstm_cell_4/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/dropout_6/CastВ
lstm_cell_4/dropout_6/Mul_1Mullstm_cell_4/dropout_6/Mul:z:0lstm_cell_4/dropout_6/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/dropout_6/Mul_1
lstm_cell_4/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2
lstm_cell_4/dropout_7/ConstЗ
lstm_cell_4/dropout_7/MulMul lstm_cell_4/ones_like_1:output:0$lstm_cell_4/dropout_7/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/dropout_7/Mul
lstm_cell_4/dropout_7/ShapeShape lstm_cell_4/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout_7/Shape§
2lstm_cell_4/dropout_7/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_7/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџd*
dtype0*
seedБџх)*
seed2Зд24
2lstm_cell_4/dropout_7/random_uniform/RandomUniform
$lstm_cell_4/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2&
$lstm_cell_4/dropout_7/GreaterEqual/yі
"lstm_cell_4/dropout_7/GreaterEqualGreaterEqual;lstm_cell_4/dropout_7/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2$
"lstm_cell_4/dropout_7/GreaterEqualЉ
lstm_cell_4/dropout_7/CastCast&lstm_cell_4/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/dropout_7/CastВ
lstm_cell_4/dropout_7/Mul_1Mullstm_cell_4/dropout_7/Mul:z:0lstm_cell_4/dropout_7/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/dropout_7/Mul_1
lstm_cell_4/mulMulstrided_slice_2:output:0lstm_cell_4/dropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/mul
lstm_cell_4/mul_1Mulstrided_slice_2:output:0lstm_cell_4/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/mul_1
lstm_cell_4/mul_2Mulstrided_slice_2:output:0lstm_cell_4/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/mul_2
lstm_cell_4/mul_3Mulstrided_slice_2:output:0lstm_cell_4/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/mul_3h
lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/Const|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dimА
 lstm_cell_4/split/ReadVariableOpReadVariableOp)lstm_cell_4_split_readvariableop_resource* 
_output_shapes
:
Ќ*
dtype02"
 lstm_cell_4/split/ReadVariableOpл
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0(lstm_cell_4/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	Ќd:	Ќd:	Ќd:	Ќd*
	num_split2
lstm_cell_4/split
lstm_cell_4/MatMulMatMullstm_cell_4/mul:z:0lstm_cell_4/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul
lstm_cell_4/MatMul_1MatMullstm_cell_4/mul_1:z:0lstm_cell_4/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul_1
lstm_cell_4/MatMul_2MatMullstm_cell_4/mul_2:z:0lstm_cell_4/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul_2
lstm_cell_4/MatMul_3MatMullstm_cell_4/mul_3:z:0lstm_cell_4/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul_3l
lstm_cell_4/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/Const_1
lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_4/split_1/split_dimБ
"lstm_cell_4/split_1/ReadVariableOpReadVariableOp+lstm_cell_4_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02$
"lstm_cell_4/split_1/ReadVariableOpЯ
lstm_cell_4/split_1Split&lstm_cell_4/split_1/split_dim:output:0*lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:d:d:d:d*
	num_split2
lstm_cell_4/split_1Ѓ
lstm_cell_4/BiasAddBiasAddlstm_cell_4/MatMul:product:0lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/BiasAddЉ
lstm_cell_4/BiasAdd_1BiasAddlstm_cell_4/MatMul_1:product:0lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/BiasAdd_1Љ
lstm_cell_4/BiasAdd_2BiasAddlstm_cell_4/MatMul_2:product:0lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/BiasAdd_2Љ
lstm_cell_4/BiasAdd_3BiasAddlstm_cell_4/MatMul_3:product:0lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/BiasAdd_3
lstm_cell_4/mul_4Mulzeros:output:0lstm_cell_4/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/mul_4
lstm_cell_4/mul_5Mulzeros:output:0lstm_cell_4/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/mul_5
lstm_cell_4/mul_6Mulzeros:output:0lstm_cell_4/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/mul_6
lstm_cell_4/mul_7Mulzeros:output:0lstm_cell_4/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/mul_7
lstm_cell_4/ReadVariableOpReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	d*
dtype02
lstm_cell_4/ReadVariableOp
lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell_4/strided_slice/stack
!lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   2#
!lstm_cell_4/strided_slice/stack_1
!lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell_4/strided_slice/stack_2Ф
lstm_cell_4/strided_sliceStridedSlice"lstm_cell_4/ReadVariableOp:value:0(lstm_cell_4/strided_slice/stack:output:0*lstm_cell_4/strided_slice/stack_1:output:0*lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2
lstm_cell_4/strided_sliceЃ
lstm_cell_4/MatMul_4MatMullstm_cell_4/mul_4:z:0"lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul_4
lstm_cell_4/addAddV2lstm_cell_4/BiasAdd:output:0lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/add|
lstm_cell_4/SigmoidSigmoidlstm_cell_4/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/SigmoidЁ
lstm_cell_4/ReadVariableOp_1ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	d*
dtype02
lstm_cell_4/ReadVariableOp_1
!lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   2#
!lstm_cell_4/strided_slice_1/stack
#lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Ш   2%
#lstm_cell_4/strided_slice_1/stack_1
#lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_4/strided_slice_1/stack_2а
lstm_cell_4/strided_slice_1StridedSlice$lstm_cell_4/ReadVariableOp_1:value:0*lstm_cell_4/strided_slice_1/stack:output:0,lstm_cell_4/strided_slice_1/stack_1:output:0,lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2
lstm_cell_4/strided_slice_1Ѕ
lstm_cell_4/MatMul_5MatMullstm_cell_4/mul_5:z:0$lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul_5Ё
lstm_cell_4/add_1AddV2lstm_cell_4/BiasAdd_1:output:0lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/add_1
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/Sigmoid_1
lstm_cell_4/mul_8Mullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/mul_8Ё
lstm_cell_4/ReadVariableOp_2ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	d*
dtype02
lstm_cell_4/ReadVariableOp_2
!lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    Ш   2#
!lstm_cell_4/strided_slice_2/stack
#lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  2%
#lstm_cell_4/strided_slice_2/stack_1
#lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_4/strided_slice_2/stack_2а
lstm_cell_4/strided_slice_2StridedSlice$lstm_cell_4/ReadVariableOp_2:value:0*lstm_cell_4/strided_slice_2/stack:output:0,lstm_cell_4/strided_slice_2/stack_1:output:0,lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2
lstm_cell_4/strided_slice_2Ѕ
lstm_cell_4/MatMul_6MatMullstm_cell_4/mul_6:z:0$lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul_6Ё
lstm_cell_4/add_2AddV2lstm_cell_4/BiasAdd_2:output:0lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/add_2u
lstm_cell_4/TanhTanhlstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/Tanh
lstm_cell_4/mul_9Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/mul_9
lstm_cell_4/add_3AddV2lstm_cell_4/mul_8:z:0lstm_cell_4/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/add_3Ё
lstm_cell_4/ReadVariableOp_3ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	d*
dtype02
lstm_cell_4/ReadVariableOp_3
!lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  2#
!lstm_cell_4/strided_slice_3/stack
#lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_4/strided_slice_3/stack_1
#lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_4/strided_slice_3/stack_2а
lstm_cell_4/strided_slice_3StridedSlice$lstm_cell_4/ReadVariableOp_3:value:0*lstm_cell_4/strided_slice_3/stack:output:0,lstm_cell_4/strided_slice_3/stack_1:output:0,lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2
lstm_cell_4/strided_slice_3Ѕ
lstm_cell_4/MatMul_7MatMullstm_cell_4/mul_7:z:0$lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul_7Ё
lstm_cell_4/add_4AddV2lstm_cell_4/BiasAdd_3:output:0lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/add_4
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Tanh_1Tanhlstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/Tanh_1
lstm_cell_4/mul_10Mullstm_cell_4/Sigmoid_2:y:0lstm_cell_4/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/mul_10
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterп
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_4_split_readvariableop_resource+lstm_cell_4_split_1_readvariableop_resource#lstm_cell_4_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_33766*
condR
while_cond_33765*K
output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџd*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџd2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:џџџџџџџџџџџџџџџџџџЌ:::2
whilewhile:_ [
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ
"
_user_specified_name
inputs/0
Ц
o
)__inference_embedding_layer_call_fn_33583

inputs
unknown
identityЂStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ(Ќ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_320182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:џџџџџџџџџ(Ќ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ(:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ(
 
_user_specified_nameinputs
№
Щ
E__inference_sequential_layer_call_and_return_conditional_losses_32746
embedding_input
embedding_32027

lstm_32690

lstm_32692

lstm_32694
dense_final_32740
dense_final_32742
identityЂ#dense_final/StatefulPartitionedCallЂ!embedding/StatefulPartitionedCallЂlstm/StatefulPartitionedCall
!embedding/StatefulPartitionedCallStatefulPartitionedCallembedding_inputembedding_32027*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ(Ќ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_320182#
!embedding/StatefulPartitionedCallЏ
lstm/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0
lstm_32690
lstm_32692
lstm_32694*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_324122
lstm/StatefulPartitionedCall
%temporal_mean_pooling/PartitionedCallPartitionedCall%lstm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_temporal_mean_pooling_layer_call_and_return_conditional_losses_327012'
%temporal_mean_pooling/PartitionedCallШ
#dense_final/StatefulPartitionedCallStatefulPartitionedCall.temporal_mean_pooling/PartitionedCall:output:0dense_final_32740dense_final_32742*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_final_layer_call_and_return_conditional_losses_327292%
#dense_final/StatefulPartitionedCallщ
IdentityIdentity,dense_final/StatefulPartitionedCall:output:0$^dense_final/StatefulPartitionedCall"^embedding/StatefulPartitionedCall^lstm/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ(::::::2J
#dense_final/StatefulPartitionedCall#dense_final/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:X T
'
_output_shapes
:џџџџџџџџџ(
)
_user_specified_nameembedding_input
ж
п
while_body_34085
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_05
1while_lstm_cell_4_split_readvariableop_resource_07
3while_lstm_cell_4_split_1_readvariableop_resource_0/
+while_lstm_cell_4_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor3
/while_lstm_cell_4_split_readvariableop_resource5
1while_lstm_cell_4_split_1_readvariableop_resource-
)while_lstm_cell_4_readvariableop_resourceУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeд
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџЌ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemІ
!while/lstm_cell_4/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/ones_like/Shape
!while/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!while/lstm_cell_4/ones_like/ConstЭ
while/lstm_cell_4/ones_likeFill*while/lstm_cell_4/ones_like/Shape:output:0*while/lstm_cell_4/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_4/ones_like
#while/lstm_cell_4/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2%
#while/lstm_cell_4/ones_like_1/Shape
#while/lstm_cell_4/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2%
#while/lstm_cell_4/ones_like_1/Constд
while/lstm_cell_4/ones_like_1Fill,while/lstm_cell_4/ones_like_1/Shape:output:0,while/lstm_cell_4/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/ones_like_1Р
while/lstm_cell_4/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_4/mulФ
while/lstm_cell_4/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_4/mul_1Ф
while/lstm_cell_4/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_4/mul_2Ф
while/lstm_cell_4/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_4/mul_3t
while/lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_4/Const
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_4/split/split_dimФ
&while/lstm_cell_4/split/ReadVariableOpReadVariableOp1while_lstm_cell_4_split_readvariableop_resource_0* 
_output_shapes
:
Ќ*
dtype02(
&while/lstm_cell_4/split/ReadVariableOpѓ
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0.while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	Ќd:	Ќd:	Ќd:	Ќd*
	num_split2
while/lstm_cell_4/split­
while/lstm_cell_4/MatMulMatMulwhile/lstm_cell_4/mul:z:0 while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMulГ
while/lstm_cell_4/MatMul_1MatMulwhile/lstm_cell_4/mul_1:z:0 while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMul_1Г
while/lstm_cell_4/MatMul_2MatMulwhile/lstm_cell_4/mul_2:z:0 while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMul_2Г
while/lstm_cell_4/MatMul_3MatMulwhile/lstm_cell_4/mul_3:z:0 while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMul_3x
while/lstm_cell_4/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_4/Const_1
#while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_4/split_1/split_dimХ
(while/lstm_cell_4/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02*
(while/lstm_cell_4/split_1/ReadVariableOpч
while/lstm_cell_4/split_1Split,while/lstm_cell_4/split_1/split_dim:output:00while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:d:d:d:d*
	num_split2
while/lstm_cell_4/split_1Л
while/lstm_cell_4/BiasAddBiasAdd"while/lstm_cell_4/MatMul:product:0"while/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/BiasAddС
while/lstm_cell_4/BiasAdd_1BiasAdd$while/lstm_cell_4/MatMul_1:product:0"while/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/BiasAdd_1С
while/lstm_cell_4/BiasAdd_2BiasAdd$while/lstm_cell_4/MatMul_2:product:0"while/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/BiasAdd_2С
while/lstm_cell_4/BiasAdd_3BiasAdd$while/lstm_cell_4/MatMul_3:product:0"while/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/BiasAdd_3Ј
while/lstm_cell_4/mul_4Mulwhile_placeholder_2&while/lstm_cell_4/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/mul_4Ј
while/lstm_cell_4/mul_5Mulwhile_placeholder_2&while/lstm_cell_4/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/mul_5Ј
while/lstm_cell_4/mul_6Mulwhile_placeholder_2&while/lstm_cell_4/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/mul_6Ј
while/lstm_cell_4/mul_7Mulwhile_placeholder_2&while/lstm_cell_4/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/mul_7Б
 while/lstm_cell_4/ReadVariableOpReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	d*
dtype02"
 while/lstm_cell_4/ReadVariableOp
%while/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell_4/strided_slice/stackЃ
'while/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   2)
'while/lstm_cell_4/strided_slice/stack_1Ѓ
'while/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell_4/strided_slice/stack_2ш
while/lstm_cell_4/strided_sliceStridedSlice(while/lstm_cell_4/ReadVariableOp:value:0.while/lstm_cell_4/strided_slice/stack:output:00while/lstm_cell_4/strided_slice/stack_1:output:00while/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2!
while/lstm_cell_4/strided_sliceЛ
while/lstm_cell_4/MatMul_4MatMulwhile/lstm_cell_4/mul_4:z:0(while/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMul_4Г
while/lstm_cell_4/addAddV2"while/lstm_cell_4/BiasAdd:output:0$while/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/add
while/lstm_cell_4/SigmoidSigmoidwhile/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/SigmoidЕ
"while/lstm_cell_4/ReadVariableOp_1ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	d*
dtype02$
"while/lstm_cell_4/ReadVariableOp_1Ѓ
'while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   2)
'while/lstm_cell_4/strided_slice_1/stackЇ
)while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Ш   2+
)while/lstm_cell_4/strided_slice_1/stack_1Ї
)while/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_4/strided_slice_1/stack_2є
!while/lstm_cell_4/strided_slice_1StridedSlice*while/lstm_cell_4/ReadVariableOp_1:value:00while/lstm_cell_4/strided_slice_1/stack:output:02while/lstm_cell_4/strided_slice_1/stack_1:output:02while/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_1Н
while/lstm_cell_4/MatMul_5MatMulwhile/lstm_cell_4/mul_5:z:0*while/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMul_5Й
while/lstm_cell_4/add_1AddV2$while/lstm_cell_4/BiasAdd_1:output:0$while/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/add_1
while/lstm_cell_4/Sigmoid_1Sigmoidwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/Sigmoid_1Ё
while/lstm_cell_4/mul_8Mulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/mul_8Е
"while/lstm_cell_4/ReadVariableOp_2ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	d*
dtype02$
"while/lstm_cell_4/ReadVariableOp_2Ѓ
'while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    Ш   2)
'while/lstm_cell_4/strided_slice_2/stackЇ
)while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  2+
)while/lstm_cell_4/strided_slice_2/stack_1Ї
)while/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_4/strided_slice_2/stack_2є
!while/lstm_cell_4/strided_slice_2StridedSlice*while/lstm_cell_4/ReadVariableOp_2:value:00while/lstm_cell_4/strided_slice_2/stack:output:02while/lstm_cell_4/strided_slice_2/stack_1:output:02while/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_2Н
while/lstm_cell_4/MatMul_6MatMulwhile/lstm_cell_4/mul_6:z:0*while/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMul_6Й
while/lstm_cell_4/add_2AddV2$while/lstm_cell_4/BiasAdd_2:output:0$while/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/add_2
while/lstm_cell_4/TanhTanhwhile/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/TanhІ
while/lstm_cell_4/mul_9Mulwhile/lstm_cell_4/Sigmoid:y:0while/lstm_cell_4/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/mul_9Ї
while/lstm_cell_4/add_3AddV2while/lstm_cell_4/mul_8:z:0while/lstm_cell_4/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/add_3Е
"while/lstm_cell_4/ReadVariableOp_3ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	d*
dtype02$
"while/lstm_cell_4/ReadVariableOp_3Ѓ
'while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  2)
'while/lstm_cell_4/strided_slice_3/stackЇ
)while/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_4/strided_slice_3/stack_1Ї
)while/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_4/strided_slice_3/stack_2є
!while/lstm_cell_4/strided_slice_3StridedSlice*while/lstm_cell_4/ReadVariableOp_3:value:00while/lstm_cell_4/strided_slice_3/stack:output:02while/lstm_cell_4/strided_slice_3/stack_1:output:02while/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_3Н
while/lstm_cell_4/MatMul_7MatMulwhile/lstm_cell_4/mul_7:z:0*while/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMul_7Й
while/lstm_cell_4/add_4AddV2$while/lstm_cell_4/BiasAdd_3:output:0$while/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/add_4
while/lstm_cell_4/Sigmoid_2Sigmoidwhile/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/Sigmoid_2
while/lstm_cell_4/Tanh_1Tanhwhile/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/Tanh_1Ќ
while/lstm_cell_4/mul_10Mulwhile/lstm_cell_4/Sigmoid_2:y:0while/lstm_cell_4/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/mul_10р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_4/mul_10:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_4/mul_10:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_4_readvariableop_resource+while_lstm_cell_4_readvariableop_resource_0"h
1while_lstm_cell_4_split_1_readvariableop_resource3while_lstm_cell_4_split_1_readvariableop_resource_0"d
/while_lstm_cell_4_split_readvariableop_resource1while_lstm_cell_4_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџd:џџџџџџџџџd: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
: 
І
О
while_cond_32530
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_32530___redundant_placeholder03
/while_while_cond_32530___redundant_placeholder13
/while_while_cond_32530___redundant_placeholder23
/while_while_cond_32530___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџd:џџџџџџџџџd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
:
Н	
Ђ
lstm_while_cond_33056&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3(
$lstm_while_less_lstm_strided_slice_1=
9lstm_while_lstm_while_cond_33056___redundant_placeholder0=
9lstm_while_lstm_while_cond_33056___redundant_placeholder1=
9lstm_while_lstm_while_cond_33056___redundant_placeholder2=
9lstm_while_lstm_while_cond_33056___redundant_placeholder3
lstm_while_identity

lstm/while/LessLesslstm_while_placeholder$lstm_while_less_lstm_strided_slice_1*
T0*
_output_shapes
: 2
lstm/while/Lessl
lstm/while/IdentityIdentitylstm/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm/while/Identity"3
lstm_while_identitylstm/while/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџd:џџџџџџџџџd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
:
йД

E__inference_sequential_layer_call_and_return_conditional_losses_33532

inputs$
 embedding_embedding_lookup_332682
.lstm_lstm_cell_4_split_readvariableop_resource4
0lstm_lstm_cell_4_split_1_readvariableop_resource,
(lstm_lstm_cell_4_readvariableop_resource.
*dense_final_matmul_readvariableop_resource/
+dense_final_biasadd_readvariableop_resource
identityЂ
lstm/whileq
embedding/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ(2
embedding/Cast
embedding/embedding_lookupResourceGather embedding_embedding_lookup_33268embedding/Cast:y:0*
Tindices0*3
_class)
'%loc:@embedding/embedding_lookup/33268*,
_output_shapes
:џџџџџџџџџ(Ќ*
dtype02
embedding/embedding_lookupч
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*3
_class)
'%loc:@embedding/embedding_lookup/33268*,
_output_shapes
:џџџџџџџџџ(Ќ2%
#embedding/embedding_lookup/IdentityП
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:џџџџџџџџџ(Ќ2'
%embedding/embedding_lookup/Identity_1v

lstm/ShapeShape.embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2

lstm/Shape~
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice/stack
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_1
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_2
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slicef
lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
lstm/zeros/mul/y
lstm/zeros/mulMullstm/strided_slice:output:0lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/muli
lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm/zeros/Less/y{
lstm/zeros/LessLesslstm/zeros/mul:z:0lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/Lessl
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
lstm/zeros/packed/1
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros/packedi
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros/Const

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2

lstm/zerosj
lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
lstm/zeros_1/mul/y
lstm/zeros_1/mulMullstm/strided_slice:output:0lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/mulm
lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm/zeros_1/Less/y
lstm/zeros_1/LessLesslstm/zeros_1/mul:z:0lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/Lessp
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
lstm/zeros_1/packed/1
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros_1/packedm
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros_1/Const
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/zeros_1
lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose/permВ
lstm/transpose	Transpose.embedding/embedding_lookup/Identity_1:output:0lstm/transpose/perm:output:0*
T0*,
_output_shapes
:(џџџџџџџџџЌ2
lstm/transpose^
lstm/Shape_1Shapelstm/transpose:y:0*
T0*
_output_shapes
:2
lstm/Shape_1
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_1/stack
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_1
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_2
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slice_1
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2"
 lstm/TensorArrayV2/element_shapeЦ
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2Щ
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  2<
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shape
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02.
,lstm/TensorArrayUnstack/TensorListFromTensor
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_2/stack
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_1
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_2
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
shrink_axis_mask2
lstm/strided_slice_2
 lstm/lstm_cell_4/ones_like/ShapeShapelstm/strided_slice_2:output:0*
T0*
_output_shapes
:2"
 lstm/lstm_cell_4/ones_like/Shape
 lstm/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2"
 lstm/lstm_cell_4/ones_like/ConstЩ
lstm/lstm_cell_4/ones_likeFill)lstm/lstm_cell_4/ones_like/Shape:output:0)lstm/lstm_cell_4/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm/lstm_cell_4/ones_like
"lstm/lstm_cell_4/ones_like_1/ShapeShapelstm/zeros:output:0*
T0*
_output_shapes
:2$
"lstm/lstm_cell_4/ones_like_1/Shape
"lstm/lstm_cell_4/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"lstm/lstm_cell_4/ones_like_1/Constа
lstm/lstm_cell_4/ones_like_1Fill+lstm/lstm_cell_4/ones_like_1/Shape:output:0+lstm/lstm_cell_4/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/ones_like_1Њ
lstm/lstm_cell_4/mulMullstm/strided_slice_2:output:0#lstm/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm/lstm_cell_4/mulЎ
lstm/lstm_cell_4/mul_1Mullstm/strided_slice_2:output:0#lstm/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm/lstm_cell_4/mul_1Ў
lstm/lstm_cell_4/mul_2Mullstm/strided_slice_2:output:0#lstm/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm/lstm_cell_4/mul_2Ў
lstm/lstm_cell_4/mul_3Mullstm/strided_slice_2:output:0#lstm/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm/lstm_cell_4/mul_3r
lstm/lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/lstm_cell_4/Const
 lstm/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 lstm/lstm_cell_4/split/split_dimП
%lstm/lstm_cell_4/split/ReadVariableOpReadVariableOp.lstm_lstm_cell_4_split_readvariableop_resource* 
_output_shapes
:
Ќ*
dtype02'
%lstm/lstm_cell_4/split/ReadVariableOpя
lstm/lstm_cell_4/splitSplit)lstm/lstm_cell_4/split/split_dim:output:0-lstm/lstm_cell_4/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	Ќd:	Ќd:	Ќd:	Ќd*
	num_split2
lstm/lstm_cell_4/splitЉ
lstm/lstm_cell_4/MatMulMatMullstm/lstm_cell_4/mul:z:0lstm/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/MatMulЏ
lstm/lstm_cell_4/MatMul_1MatMullstm/lstm_cell_4/mul_1:z:0lstm/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/MatMul_1Џ
lstm/lstm_cell_4/MatMul_2MatMullstm/lstm_cell_4/mul_2:z:0lstm/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/MatMul_2Џ
lstm/lstm_cell_4/MatMul_3MatMullstm/lstm_cell_4/mul_3:z:0lstm/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/MatMul_3v
lstm/lstm_cell_4/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm/lstm_cell_4/Const_1
"lstm/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"lstm/lstm_cell_4/split_1/split_dimР
'lstm/lstm_cell_4/split_1/ReadVariableOpReadVariableOp0lstm_lstm_cell_4_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02)
'lstm/lstm_cell_4/split_1/ReadVariableOpу
lstm/lstm_cell_4/split_1Split+lstm/lstm_cell_4/split_1/split_dim:output:0/lstm/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:d:d:d:d*
	num_split2
lstm/lstm_cell_4/split_1З
lstm/lstm_cell_4/BiasAddBiasAdd!lstm/lstm_cell_4/MatMul:product:0!lstm/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/BiasAddН
lstm/lstm_cell_4/BiasAdd_1BiasAdd#lstm/lstm_cell_4/MatMul_1:product:0!lstm/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/BiasAdd_1Н
lstm/lstm_cell_4/BiasAdd_2BiasAdd#lstm/lstm_cell_4/MatMul_2:product:0!lstm/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/BiasAdd_2Н
lstm/lstm_cell_4/BiasAdd_3BiasAdd#lstm/lstm_cell_4/MatMul_3:product:0!lstm/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/BiasAdd_3Ѕ
lstm/lstm_cell_4/mul_4Mullstm/zeros:output:0%lstm/lstm_cell_4/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/mul_4Ѕ
lstm/lstm_cell_4/mul_5Mullstm/zeros:output:0%lstm/lstm_cell_4/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/mul_5Ѕ
lstm/lstm_cell_4/mul_6Mullstm/zeros:output:0%lstm/lstm_cell_4/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/mul_6Ѕ
lstm/lstm_cell_4/mul_7Mullstm/zeros:output:0%lstm/lstm_cell_4/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/mul_7Ќ
lstm/lstm_cell_4/ReadVariableOpReadVariableOp(lstm_lstm_cell_4_readvariableop_resource*
_output_shapes
:	d*
dtype02!
lstm/lstm_cell_4/ReadVariableOp
$lstm/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm/lstm_cell_4/strided_slice/stackЁ
&lstm/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   2(
&lstm/lstm_cell_4/strided_slice/stack_1Ё
&lstm/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&lstm/lstm_cell_4/strided_slice/stack_2т
lstm/lstm_cell_4/strided_sliceStridedSlice'lstm/lstm_cell_4/ReadVariableOp:value:0-lstm/lstm_cell_4/strided_slice/stack:output:0/lstm/lstm_cell_4/strided_slice/stack_1:output:0/lstm/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2 
lstm/lstm_cell_4/strided_sliceЗ
lstm/lstm_cell_4/MatMul_4MatMullstm/lstm_cell_4/mul_4:z:0'lstm/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/MatMul_4Џ
lstm/lstm_cell_4/addAddV2!lstm/lstm_cell_4/BiasAdd:output:0#lstm/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/add
lstm/lstm_cell_4/SigmoidSigmoidlstm/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/SigmoidА
!lstm/lstm_cell_4/ReadVariableOp_1ReadVariableOp(lstm_lstm_cell_4_readvariableop_resource*
_output_shapes
:	d*
dtype02#
!lstm/lstm_cell_4/ReadVariableOp_1Ё
&lstm/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   2(
&lstm/lstm_cell_4/strided_slice_1/stackЅ
(lstm/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Ш   2*
(lstm/lstm_cell_4/strided_slice_1/stack_1Ѕ
(lstm/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(lstm/lstm_cell_4/strided_slice_1/stack_2ю
 lstm/lstm_cell_4/strided_slice_1StridedSlice)lstm/lstm_cell_4/ReadVariableOp_1:value:0/lstm/lstm_cell_4/strided_slice_1/stack:output:01lstm/lstm_cell_4/strided_slice_1/stack_1:output:01lstm/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2"
 lstm/lstm_cell_4/strided_slice_1Й
lstm/lstm_cell_4/MatMul_5MatMullstm/lstm_cell_4/mul_5:z:0)lstm/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/MatMul_5Е
lstm/lstm_cell_4/add_1AddV2#lstm/lstm_cell_4/BiasAdd_1:output:0#lstm/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/add_1
lstm/lstm_cell_4/Sigmoid_1Sigmoidlstm/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/Sigmoid_1 
lstm/lstm_cell_4/mul_8Mullstm/lstm_cell_4/Sigmoid_1:y:0lstm/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/mul_8А
!lstm/lstm_cell_4/ReadVariableOp_2ReadVariableOp(lstm_lstm_cell_4_readvariableop_resource*
_output_shapes
:	d*
dtype02#
!lstm/lstm_cell_4/ReadVariableOp_2Ё
&lstm/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    Ш   2(
&lstm/lstm_cell_4/strided_slice_2/stackЅ
(lstm/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  2*
(lstm/lstm_cell_4/strided_slice_2/stack_1Ѕ
(lstm/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(lstm/lstm_cell_4/strided_slice_2/stack_2ю
 lstm/lstm_cell_4/strided_slice_2StridedSlice)lstm/lstm_cell_4/ReadVariableOp_2:value:0/lstm/lstm_cell_4/strided_slice_2/stack:output:01lstm/lstm_cell_4/strided_slice_2/stack_1:output:01lstm/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2"
 lstm/lstm_cell_4/strided_slice_2Й
lstm/lstm_cell_4/MatMul_6MatMullstm/lstm_cell_4/mul_6:z:0)lstm/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/MatMul_6Е
lstm/lstm_cell_4/add_2AddV2#lstm/lstm_cell_4/BiasAdd_2:output:0#lstm/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/add_2
lstm/lstm_cell_4/TanhTanhlstm/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/TanhЂ
lstm/lstm_cell_4/mul_9Mullstm/lstm_cell_4/Sigmoid:y:0lstm/lstm_cell_4/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/mul_9Ѓ
lstm/lstm_cell_4/add_3AddV2lstm/lstm_cell_4/mul_8:z:0lstm/lstm_cell_4/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/add_3А
!lstm/lstm_cell_4/ReadVariableOp_3ReadVariableOp(lstm_lstm_cell_4_readvariableop_resource*
_output_shapes
:	d*
dtype02#
!lstm/lstm_cell_4/ReadVariableOp_3Ё
&lstm/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  2(
&lstm/lstm_cell_4/strided_slice_3/stackЅ
(lstm/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(lstm/lstm_cell_4/strided_slice_3/stack_1Ѕ
(lstm/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(lstm/lstm_cell_4/strided_slice_3/stack_2ю
 lstm/lstm_cell_4/strided_slice_3StridedSlice)lstm/lstm_cell_4/ReadVariableOp_3:value:0/lstm/lstm_cell_4/strided_slice_3/stack:output:01lstm/lstm_cell_4/strided_slice_3/stack_1:output:01lstm/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2"
 lstm/lstm_cell_4/strided_slice_3Й
lstm/lstm_cell_4/MatMul_7MatMullstm/lstm_cell_4/mul_7:z:0)lstm/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/MatMul_7Е
lstm/lstm_cell_4/add_4AddV2#lstm/lstm_cell_4/BiasAdd_3:output:0#lstm/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/add_4
lstm/lstm_cell_4/Sigmoid_2Sigmoidlstm/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/Sigmoid_2
lstm/lstm_cell_4/Tanh_1Tanhlstm/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/Tanh_1Ј
lstm/lstm_cell_4/mul_10Mullstm/lstm_cell_4/Sigmoid_2:y:0lstm/lstm_cell_4/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/lstm_cell_4/mul_10
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   2$
"lstm/TensorArrayV2_1/element_shapeЬ
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2_1X
	lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
	lstm/time
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
lstm/while/maximum_iterationst
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm/while/loop_counterЊ

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0.lstm_lstm_cell_4_split_readvariableop_resource0lstm_lstm_cell_4_split_1_readvariableop_resource(lstm_lstm_cell_4_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *%
_read_only_resource_inputs
	
*!
bodyR
lstm_while_body_33389*!
condR
lstm_while_cond_33388*K
output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *
parallel_iterations 2

lstm/whileП
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   27
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeќ
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:(џџџџџџџџџd*
element_dtype02)
'lstm/TensorArrayV2Stack/TensorListStack
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
lstm/strided_slice_3/stack
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_3/stack_1
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_3/stack_2И
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_mask2
lstm/strided_slice_3
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose_1/permЙ
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ(d2
lstm/transpose_1p
lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/runtimeБ
!dense_final/MatMul/ReadVariableOpReadVariableOp*dense_final_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02#
!dense_final/MatMul/ReadVariableOpЎ
dense_final/MatMulMatMullstm/strided_slice_3:output:0)dense_final/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_final/MatMulА
"dense_final/BiasAdd/ReadVariableOpReadVariableOp+dense_final_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"dense_final/BiasAdd/ReadVariableOpБ
dense_final/BiasAddBiasAdddense_final/MatMul:product:0*dense_final/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_final/BiasAdd
dense_final/SoftmaxSoftmaxdense_final/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_final/Softmax~
IdentityIdentitydense_final/Softmax:softmax:0^lstm/while*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ(::::::2

lstm/while
lstm/while:O K
'
_output_shapes
:џџџџџџџџџ(
 
_user_specified_nameinputs
§

$__inference_lstm_layer_call_fn_34892

inputs
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCallќ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_324122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:џџџџџџџџџ(Ќ:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ(Ќ
 
_user_specified_nameinputs
І
О
while_cond_34084
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_34084___redundant_placeholder03
/while_while_cond_34084___redundant_placeholder13
/while_while_cond_34084___redundant_placeholder23
/while_while_cond_34084___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџd:џџџџџџџџџd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
:

ў
 sequential_lstm_while_cond_31084<
8sequential_lstm_while_sequential_lstm_while_loop_counterB
>sequential_lstm_while_sequential_lstm_while_maximum_iterations%
!sequential_lstm_while_placeholder'
#sequential_lstm_while_placeholder_1'
#sequential_lstm_while_placeholder_2'
#sequential_lstm_while_placeholder_3>
:sequential_lstm_while_less_sequential_lstm_strided_slice_1S
Osequential_lstm_while_sequential_lstm_while_cond_31084___redundant_placeholder0S
Osequential_lstm_while_sequential_lstm_while_cond_31084___redundant_placeholder1S
Osequential_lstm_while_sequential_lstm_while_cond_31084___redundant_placeholder2S
Osequential_lstm_while_sequential_lstm_while_cond_31084___redundant_placeholder3"
sequential_lstm_while_identity
Р
sequential/lstm/while/LessLess!sequential_lstm_while_placeholder:sequential_lstm_while_less_sequential_lstm_strided_slice_1*
T0*
_output_shapes
: 2
sequential/lstm/while/Less
sequential/lstm/while/IdentityIdentitysequential/lstm/while/Less:z:0*
T0
*
_output_shapes
: 2 
sequential/lstm/while/Identity"I
sequential_lstm_while_identity'sequential/lstm/while/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџd:џџџџџџџџџd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
:
пљ
№
lstm_while_body_33057&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3%
!lstm_while_lstm_strided_slice_1_0a
]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0:
6lstm_while_lstm_cell_4_split_readvariableop_resource_0<
8lstm_while_lstm_cell_4_split_1_readvariableop_resource_04
0lstm_while_lstm_cell_4_readvariableop_resource_0
lstm_while_identity
lstm_while_identity_1
lstm_while_identity_2
lstm_while_identity_3
lstm_while_identity_4
lstm_while_identity_5#
lstm_while_lstm_strided_slice_1_
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor8
4lstm_while_lstm_cell_4_split_readvariableop_resource:
6lstm_while_lstm_cell_4_split_1_readvariableop_resource2
.lstm_while_lstm_cell_4_readvariableop_resourceЭ
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  2>
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeђ
.lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0lstm_while_placeholderElstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџЌ*
element_dtype020
.lstm/while/TensorArrayV2Read/TensorListGetItemЕ
&lstm/while/lstm_cell_4/ones_like/ShapeShape5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2(
&lstm/while/lstm_cell_4/ones_like/Shape
&lstm/while/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2(
&lstm/while/lstm_cell_4/ones_like/Constс
 lstm/while/lstm_cell_4/ones_likeFill/lstm/while/lstm_cell_4/ones_like/Shape:output:0/lstm/while/lstm_cell_4/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2"
 lstm/while/lstm_cell_4/ones_like
$lstm/while/lstm_cell_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2&
$lstm/while/lstm_cell_4/dropout/Constм
"lstm/while/lstm_cell_4/dropout/MulMul)lstm/while/lstm_cell_4/ones_like:output:0-lstm/while/lstm_cell_4/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2$
"lstm/while/lstm_cell_4/dropout/MulЅ
$lstm/while/lstm_cell_4/dropout/ShapeShape)lstm/while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2&
$lstm/while/lstm_cell_4/dropout/Shape
;lstm/while/lstm_cell_4/dropout/random_uniform/RandomUniformRandomUniform-lstm/while/lstm_cell_4/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*
seedБџх)*
seed2сТЮ2=
;lstm/while/lstm_cell_4/dropout/random_uniform/RandomUniformЃ
-lstm/while/lstm_cell_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2/
-lstm/while/lstm_cell_4/dropout/GreaterEqual/y
+lstm/while/lstm_cell_4/dropout/GreaterEqualGreaterEqualDlstm/while/lstm_cell_4/dropout/random_uniform/RandomUniform:output:06lstm/while/lstm_cell_4/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2-
+lstm/while/lstm_cell_4/dropout/GreaterEqualХ
#lstm/while/lstm_cell_4/dropout/CastCast/lstm/while/lstm_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2%
#lstm/while/lstm_cell_4/dropout/Castз
$lstm/while/lstm_cell_4/dropout/Mul_1Mul&lstm/while/lstm_cell_4/dropout/Mul:z:0'lstm/while/lstm_cell_4/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2&
$lstm/while/lstm_cell_4/dropout/Mul_1
&lstm/while/lstm_cell_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2(
&lstm/while/lstm_cell_4/dropout_1/Constт
$lstm/while/lstm_cell_4/dropout_1/MulMul)lstm/while/lstm_cell_4/ones_like:output:0/lstm/while/lstm_cell_4/dropout_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2&
$lstm/while/lstm_cell_4/dropout_1/MulЉ
&lstm/while/lstm_cell_4/dropout_1/ShapeShape)lstm/while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2(
&lstm/while/lstm_cell_4/dropout_1/Shape
=lstm/while/lstm_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform/lstm/while/lstm_cell_4/dropout_1/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*
seedБџх)*
seed2Јьд2?
=lstm/while/lstm_cell_4/dropout_1/random_uniform/RandomUniformЇ
/lstm/while/lstm_cell_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?21
/lstm/while/lstm_cell_4/dropout_1/GreaterEqual/yЃ
-lstm/while/lstm_cell_4/dropout_1/GreaterEqualGreaterEqualFlstm/while/lstm_cell_4/dropout_1/random_uniform/RandomUniform:output:08lstm/while/lstm_cell_4/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2/
-lstm/while/lstm_cell_4/dropout_1/GreaterEqualЫ
%lstm/while/lstm_cell_4/dropout_1/CastCast1lstm/while/lstm_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2'
%lstm/while/lstm_cell_4/dropout_1/Castп
&lstm/while/lstm_cell_4/dropout_1/Mul_1Mul(lstm/while/lstm_cell_4/dropout_1/Mul:z:0)lstm/while/lstm_cell_4/dropout_1/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2(
&lstm/while/lstm_cell_4/dropout_1/Mul_1
&lstm/while/lstm_cell_4/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2(
&lstm/while/lstm_cell_4/dropout_2/Constт
$lstm/while/lstm_cell_4/dropout_2/MulMul)lstm/while/lstm_cell_4/ones_like:output:0/lstm/while/lstm_cell_4/dropout_2/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2&
$lstm/while/lstm_cell_4/dropout_2/MulЉ
&lstm/while/lstm_cell_4/dropout_2/ShapeShape)lstm/while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2(
&lstm/while/lstm_cell_4/dropout_2/Shape
=lstm/while/lstm_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform/lstm/while/lstm_cell_4/dropout_2/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*
seedБџх)*
seed2ЭЪ42?
=lstm/while/lstm_cell_4/dropout_2/random_uniform/RandomUniformЇ
/lstm/while/lstm_cell_4/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?21
/lstm/while/lstm_cell_4/dropout_2/GreaterEqual/yЃ
-lstm/while/lstm_cell_4/dropout_2/GreaterEqualGreaterEqualFlstm/while/lstm_cell_4/dropout_2/random_uniform/RandomUniform:output:08lstm/while/lstm_cell_4/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2/
-lstm/while/lstm_cell_4/dropout_2/GreaterEqualЫ
%lstm/while/lstm_cell_4/dropout_2/CastCast1lstm/while/lstm_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2'
%lstm/while/lstm_cell_4/dropout_2/Castп
&lstm/while/lstm_cell_4/dropout_2/Mul_1Mul(lstm/while/lstm_cell_4/dropout_2/Mul:z:0)lstm/while/lstm_cell_4/dropout_2/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2(
&lstm/while/lstm_cell_4/dropout_2/Mul_1
&lstm/while/lstm_cell_4/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2(
&lstm/while/lstm_cell_4/dropout_3/Constт
$lstm/while/lstm_cell_4/dropout_3/MulMul)lstm/while/lstm_cell_4/ones_like:output:0/lstm/while/lstm_cell_4/dropout_3/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2&
$lstm/while/lstm_cell_4/dropout_3/MulЉ
&lstm/while/lstm_cell_4/dropout_3/ShapeShape)lstm/while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2(
&lstm/while/lstm_cell_4/dropout_3/Shape
=lstm/while/lstm_cell_4/dropout_3/random_uniform/RandomUniformRandomUniform/lstm/while/lstm_cell_4/dropout_3/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*
seedБџх)*
seed2ЦЇ2?
=lstm/while/lstm_cell_4/dropout_3/random_uniform/RandomUniformЇ
/lstm/while/lstm_cell_4/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?21
/lstm/while/lstm_cell_4/dropout_3/GreaterEqual/yЃ
-lstm/while/lstm_cell_4/dropout_3/GreaterEqualGreaterEqualFlstm/while/lstm_cell_4/dropout_3/random_uniform/RandomUniform:output:08lstm/while/lstm_cell_4/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2/
-lstm/while/lstm_cell_4/dropout_3/GreaterEqualЫ
%lstm/while/lstm_cell_4/dropout_3/CastCast1lstm/while/lstm_cell_4/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2'
%lstm/while/lstm_cell_4/dropout_3/Castп
&lstm/while/lstm_cell_4/dropout_3/Mul_1Mul(lstm/while/lstm_cell_4/dropout_3/Mul:z:0)lstm/while/lstm_cell_4/dropout_3/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2(
&lstm/while/lstm_cell_4/dropout_3/Mul_1
(lstm/while/lstm_cell_4/ones_like_1/ShapeShapelstm_while_placeholder_2*
T0*
_output_shapes
:2*
(lstm/while/lstm_cell_4/ones_like_1/Shape
(lstm/while/lstm_cell_4/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2*
(lstm/while/lstm_cell_4/ones_like_1/Constш
"lstm/while/lstm_cell_4/ones_like_1Fill1lstm/while/lstm_cell_4/ones_like_1/Shape:output:01lstm/while/lstm_cell_4/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2$
"lstm/while/lstm_cell_4/ones_like_1
&lstm/while/lstm_cell_4/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2(
&lstm/while/lstm_cell_4/dropout_4/Constу
$lstm/while/lstm_cell_4/dropout_4/MulMul+lstm/while/lstm_cell_4/ones_like_1:output:0/lstm/while/lstm_cell_4/dropout_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2&
$lstm/while/lstm_cell_4/dropout_4/MulЋ
&lstm/while/lstm_cell_4/dropout_4/ShapeShape+lstm/while/lstm_cell_4/ones_like_1:output:0*
T0*
_output_shapes
:2(
&lstm/while/lstm_cell_4/dropout_4/Shape
=lstm/while/lstm_cell_4/dropout_4/random_uniform/RandomUniformRandomUniform/lstm/while/lstm_cell_4/dropout_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџd*
dtype0*
seedБџх)*
seed2Џ2?
=lstm/while/lstm_cell_4/dropout_4/random_uniform/RandomUniformЇ
/lstm/while/lstm_cell_4/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?21
/lstm/while/lstm_cell_4/dropout_4/GreaterEqual/yЂ
-lstm/while/lstm_cell_4/dropout_4/GreaterEqualGreaterEqualFlstm/while/lstm_cell_4/dropout_4/random_uniform/RandomUniform:output:08lstm/while/lstm_cell_4/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2/
-lstm/while/lstm_cell_4/dropout_4/GreaterEqualЪ
%lstm/while/lstm_cell_4/dropout_4/CastCast1lstm/while/lstm_cell_4/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd2'
%lstm/while/lstm_cell_4/dropout_4/Castо
&lstm/while/lstm_cell_4/dropout_4/Mul_1Mul(lstm/while/lstm_cell_4/dropout_4/Mul:z:0)lstm/while/lstm_cell_4/dropout_4/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2(
&lstm/while/lstm_cell_4/dropout_4/Mul_1
&lstm/while/lstm_cell_4/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2(
&lstm/while/lstm_cell_4/dropout_5/Constу
$lstm/while/lstm_cell_4/dropout_5/MulMul+lstm/while/lstm_cell_4/ones_like_1:output:0/lstm/while/lstm_cell_4/dropout_5/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2&
$lstm/while/lstm_cell_4/dropout_5/MulЋ
&lstm/while/lstm_cell_4/dropout_5/ShapeShape+lstm/while/lstm_cell_4/ones_like_1:output:0*
T0*
_output_shapes
:2(
&lstm/while/lstm_cell_4/dropout_5/Shape
=lstm/while/lstm_cell_4/dropout_5/random_uniform/RandomUniformRandomUniform/lstm/while/lstm_cell_4/dropout_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџd*
dtype0*
seedБџх)*
seed2ЕьД2?
=lstm/while/lstm_cell_4/dropout_5/random_uniform/RandomUniformЇ
/lstm/while/lstm_cell_4/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?21
/lstm/while/lstm_cell_4/dropout_5/GreaterEqual/yЂ
-lstm/while/lstm_cell_4/dropout_5/GreaterEqualGreaterEqualFlstm/while/lstm_cell_4/dropout_5/random_uniform/RandomUniform:output:08lstm/while/lstm_cell_4/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2/
-lstm/while/lstm_cell_4/dropout_5/GreaterEqualЪ
%lstm/while/lstm_cell_4/dropout_5/CastCast1lstm/while/lstm_cell_4/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd2'
%lstm/while/lstm_cell_4/dropout_5/Castо
&lstm/while/lstm_cell_4/dropout_5/Mul_1Mul(lstm/while/lstm_cell_4/dropout_5/Mul:z:0)lstm/while/lstm_cell_4/dropout_5/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2(
&lstm/while/lstm_cell_4/dropout_5/Mul_1
&lstm/while/lstm_cell_4/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2(
&lstm/while/lstm_cell_4/dropout_6/Constу
$lstm/while/lstm_cell_4/dropout_6/MulMul+lstm/while/lstm_cell_4/ones_like_1:output:0/lstm/while/lstm_cell_4/dropout_6/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2&
$lstm/while/lstm_cell_4/dropout_6/MulЋ
&lstm/while/lstm_cell_4/dropout_6/ShapeShape+lstm/while/lstm_cell_4/ones_like_1:output:0*
T0*
_output_shapes
:2(
&lstm/while/lstm_cell_4/dropout_6/Shape
=lstm/while/lstm_cell_4/dropout_6/random_uniform/RandomUniformRandomUniform/lstm/while/lstm_cell_4/dropout_6/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџd*
dtype0*
seedБџх)*
seed2нѓ2?
=lstm/while/lstm_cell_4/dropout_6/random_uniform/RandomUniformЇ
/lstm/while/lstm_cell_4/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?21
/lstm/while/lstm_cell_4/dropout_6/GreaterEqual/yЂ
-lstm/while/lstm_cell_4/dropout_6/GreaterEqualGreaterEqualFlstm/while/lstm_cell_4/dropout_6/random_uniform/RandomUniform:output:08lstm/while/lstm_cell_4/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2/
-lstm/while/lstm_cell_4/dropout_6/GreaterEqualЪ
%lstm/while/lstm_cell_4/dropout_6/CastCast1lstm/while/lstm_cell_4/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd2'
%lstm/while/lstm_cell_4/dropout_6/Castо
&lstm/while/lstm_cell_4/dropout_6/Mul_1Mul(lstm/while/lstm_cell_4/dropout_6/Mul:z:0)lstm/while/lstm_cell_4/dropout_6/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2(
&lstm/while/lstm_cell_4/dropout_6/Mul_1
&lstm/while/lstm_cell_4/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2(
&lstm/while/lstm_cell_4/dropout_7/Constу
$lstm/while/lstm_cell_4/dropout_7/MulMul+lstm/while/lstm_cell_4/ones_like_1:output:0/lstm/while/lstm_cell_4/dropout_7/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2&
$lstm/while/lstm_cell_4/dropout_7/MulЋ
&lstm/while/lstm_cell_4/dropout_7/ShapeShape+lstm/while/lstm_cell_4/ones_like_1:output:0*
T0*
_output_shapes
:2(
&lstm/while/lstm_cell_4/dropout_7/Shape
=lstm/while/lstm_cell_4/dropout_7/random_uniform/RandomUniformRandomUniform/lstm/while/lstm_cell_4/dropout_7/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџd*
dtype0*
seedБџх)*
seed2яЂ2?
=lstm/while/lstm_cell_4/dropout_7/random_uniform/RandomUniformЇ
/lstm/while/lstm_cell_4/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?21
/lstm/while/lstm_cell_4/dropout_7/GreaterEqual/yЂ
-lstm/while/lstm_cell_4/dropout_7/GreaterEqualGreaterEqualFlstm/while/lstm_cell_4/dropout_7/random_uniform/RandomUniform:output:08lstm/while/lstm_cell_4/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2/
-lstm/while/lstm_cell_4/dropout_7/GreaterEqualЪ
%lstm/while/lstm_cell_4/dropout_7/CastCast1lstm/while/lstm_cell_4/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd2'
%lstm/while/lstm_cell_4/dropout_7/Castо
&lstm/while/lstm_cell_4/dropout_7/Mul_1Mul(lstm/while/lstm_cell_4/dropout_7/Mul:z:0)lstm/while/lstm_cell_4/dropout_7/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2(
&lstm/while/lstm_cell_4/dropout_7/Mul_1г
lstm/while/lstm_cell_4/mulMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0(lstm/while/lstm_cell_4/dropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm/while/lstm_cell_4/mulй
lstm/while/lstm_cell_4/mul_1Mul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0*lstm/while/lstm_cell_4/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm/while/lstm_cell_4/mul_1й
lstm/while/lstm_cell_4/mul_2Mul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0*lstm/while/lstm_cell_4/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm/while/lstm_cell_4/mul_2й
lstm/while/lstm_cell_4/mul_3Mul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0*lstm/while/lstm_cell_4/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm/while/lstm_cell_4/mul_3~
lstm/while/lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/lstm_cell_4/Const
&lstm/while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2(
&lstm/while/lstm_cell_4/split/split_dimг
+lstm/while/lstm_cell_4/split/ReadVariableOpReadVariableOp6lstm_while_lstm_cell_4_split_readvariableop_resource_0* 
_output_shapes
:
Ќ*
dtype02-
+lstm/while/lstm_cell_4/split/ReadVariableOp
lstm/while/lstm_cell_4/splitSplit/lstm/while/lstm_cell_4/split/split_dim:output:03lstm/while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	Ќd:	Ќd:	Ќd:	Ќd*
	num_split2
lstm/while/lstm_cell_4/splitС
lstm/while/lstm_cell_4/MatMulMatMullstm/while/lstm_cell_4/mul:z:0%lstm/while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/while/lstm_cell_4/MatMulЧ
lstm/while/lstm_cell_4/MatMul_1MatMul lstm/while/lstm_cell_4/mul_1:z:0%lstm/while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd2!
lstm/while/lstm_cell_4/MatMul_1Ч
lstm/while/lstm_cell_4/MatMul_2MatMul lstm/while/lstm_cell_4/mul_2:z:0%lstm/while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd2!
lstm/while/lstm_cell_4/MatMul_2Ч
lstm/while/lstm_cell_4/MatMul_3MatMul lstm/while/lstm_cell_4/mul_3:z:0%lstm/while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџd2!
lstm/while/lstm_cell_4/MatMul_3
lstm/while/lstm_cell_4/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2 
lstm/while/lstm_cell_4/Const_1
(lstm/while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(lstm/while/lstm_cell_4/split_1/split_dimд
-lstm/while/lstm_cell_4/split_1/ReadVariableOpReadVariableOp8lstm_while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02/
-lstm/while/lstm_cell_4/split_1/ReadVariableOpћ
lstm/while/lstm_cell_4/split_1Split1lstm/while/lstm_cell_4/split_1/split_dim:output:05lstm/while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:d:d:d:d*
	num_split2 
lstm/while/lstm_cell_4/split_1Я
lstm/while/lstm_cell_4/BiasAddBiasAdd'lstm/while/lstm_cell_4/MatMul:product:0'lstm/while/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2 
lstm/while/lstm_cell_4/BiasAddе
 lstm/while/lstm_cell_4/BiasAdd_1BiasAdd)lstm/while/lstm_cell_4/MatMul_1:product:0'lstm/while/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџd2"
 lstm/while/lstm_cell_4/BiasAdd_1е
 lstm/while/lstm_cell_4/BiasAdd_2BiasAdd)lstm/while/lstm_cell_4/MatMul_2:product:0'lstm/while/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџd2"
 lstm/while/lstm_cell_4/BiasAdd_2е
 lstm/while/lstm_cell_4/BiasAdd_3BiasAdd)lstm/while/lstm_cell_4/MatMul_3:product:0'lstm/while/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџd2"
 lstm/while/lstm_cell_4/BiasAdd_3Л
lstm/while/lstm_cell_4/mul_4Mullstm_while_placeholder_2*lstm/while/lstm_cell_4/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/while/lstm_cell_4/mul_4Л
lstm/while/lstm_cell_4/mul_5Mullstm_while_placeholder_2*lstm/while/lstm_cell_4/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/while/lstm_cell_4/mul_5Л
lstm/while/lstm_cell_4/mul_6Mullstm_while_placeholder_2*lstm/while/lstm_cell_4/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/while/lstm_cell_4/mul_6Л
lstm/while/lstm_cell_4/mul_7Mullstm_while_placeholder_2*lstm/while/lstm_cell_4/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/while/lstm_cell_4/mul_7Р
%lstm/while/lstm_cell_4/ReadVariableOpReadVariableOp0lstm_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	d*
dtype02'
%lstm/while/lstm_cell_4/ReadVariableOpЉ
*lstm/while/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2,
*lstm/while/lstm_cell_4/strided_slice/stack­
,lstm/while/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   2.
,lstm/while/lstm_cell_4/strided_slice/stack_1­
,lstm/while/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm/while/lstm_cell_4/strided_slice/stack_2
$lstm/while/lstm_cell_4/strided_sliceStridedSlice-lstm/while/lstm_cell_4/ReadVariableOp:value:03lstm/while/lstm_cell_4/strided_slice/stack:output:05lstm/while/lstm_cell_4/strided_slice/stack_1:output:05lstm/while/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2&
$lstm/while/lstm_cell_4/strided_sliceЯ
lstm/while/lstm_cell_4/MatMul_4MatMul lstm/while/lstm_cell_4/mul_4:z:0-lstm/while/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2!
lstm/while/lstm_cell_4/MatMul_4Ч
lstm/while/lstm_cell_4/addAddV2'lstm/while/lstm_cell_4/BiasAdd:output:0)lstm/while/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/while/lstm_cell_4/add
lstm/while/lstm_cell_4/SigmoidSigmoidlstm/while/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2 
lstm/while/lstm_cell_4/SigmoidФ
'lstm/while/lstm_cell_4/ReadVariableOp_1ReadVariableOp0lstm_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	d*
dtype02)
'lstm/while/lstm_cell_4/ReadVariableOp_1­
,lstm/while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   2.
,lstm/while/lstm_cell_4/strided_slice_1/stackБ
.lstm/while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Ш   20
.lstm/while/lstm_cell_4/strided_slice_1/stack_1Б
.lstm/while/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.lstm/while/lstm_cell_4/strided_slice_1/stack_2
&lstm/while/lstm_cell_4/strided_slice_1StridedSlice/lstm/while/lstm_cell_4/ReadVariableOp_1:value:05lstm/while/lstm_cell_4/strided_slice_1/stack:output:07lstm/while/lstm_cell_4/strided_slice_1/stack_1:output:07lstm/while/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2(
&lstm/while/lstm_cell_4/strided_slice_1б
lstm/while/lstm_cell_4/MatMul_5MatMul lstm/while/lstm_cell_4/mul_5:z:0/lstm/while/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2!
lstm/while/lstm_cell_4/MatMul_5Э
lstm/while/lstm_cell_4/add_1AddV2)lstm/while/lstm_cell_4/BiasAdd_1:output:0)lstm/while/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/while/lstm_cell_4/add_1Ѓ
 lstm/while/lstm_cell_4/Sigmoid_1Sigmoid lstm/while/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2"
 lstm/while/lstm_cell_4/Sigmoid_1Е
lstm/while/lstm_cell_4/mul_8Mul$lstm/while/lstm_cell_4/Sigmoid_1:y:0lstm_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/while/lstm_cell_4/mul_8Ф
'lstm/while/lstm_cell_4/ReadVariableOp_2ReadVariableOp0lstm_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	d*
dtype02)
'lstm/while/lstm_cell_4/ReadVariableOp_2­
,lstm/while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    Ш   2.
,lstm/while/lstm_cell_4/strided_slice_2/stackБ
.lstm/while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  20
.lstm/while/lstm_cell_4/strided_slice_2/stack_1Б
.lstm/while/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.lstm/while/lstm_cell_4/strided_slice_2/stack_2
&lstm/while/lstm_cell_4/strided_slice_2StridedSlice/lstm/while/lstm_cell_4/ReadVariableOp_2:value:05lstm/while/lstm_cell_4/strided_slice_2/stack:output:07lstm/while/lstm_cell_4/strided_slice_2/stack_1:output:07lstm/while/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2(
&lstm/while/lstm_cell_4/strided_slice_2б
lstm/while/lstm_cell_4/MatMul_6MatMul lstm/while/lstm_cell_4/mul_6:z:0/lstm/while/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2!
lstm/while/lstm_cell_4/MatMul_6Э
lstm/while/lstm_cell_4/add_2AddV2)lstm/while/lstm_cell_4/BiasAdd_2:output:0)lstm/while/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/while/lstm_cell_4/add_2
lstm/while/lstm_cell_4/TanhTanh lstm/while/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/while/lstm_cell_4/TanhК
lstm/while/lstm_cell_4/mul_9Mul"lstm/while/lstm_cell_4/Sigmoid:y:0lstm/while/lstm_cell_4/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/while/lstm_cell_4/mul_9Л
lstm/while/lstm_cell_4/add_3AddV2 lstm/while/lstm_cell_4/mul_8:z:0 lstm/while/lstm_cell_4/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/while/lstm_cell_4/add_3Ф
'lstm/while/lstm_cell_4/ReadVariableOp_3ReadVariableOp0lstm_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	d*
dtype02)
'lstm/while/lstm_cell_4/ReadVariableOp_3­
,lstm/while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  2.
,lstm/while/lstm_cell_4/strided_slice_3/stackБ
.lstm/while/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.lstm/while/lstm_cell_4/strided_slice_3/stack_1Б
.lstm/while/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.lstm/while/lstm_cell_4/strided_slice_3/stack_2
&lstm/while/lstm_cell_4/strided_slice_3StridedSlice/lstm/while/lstm_cell_4/ReadVariableOp_3:value:05lstm/while/lstm_cell_4/strided_slice_3/stack:output:07lstm/while/lstm_cell_4/strided_slice_3/stack_1:output:07lstm/while/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2(
&lstm/while/lstm_cell_4/strided_slice_3б
lstm/while/lstm_cell_4/MatMul_7MatMul lstm/while/lstm_cell_4/mul_7:z:0/lstm/while/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2!
lstm/while/lstm_cell_4/MatMul_7Э
lstm/while/lstm_cell_4/add_4AddV2)lstm/while/lstm_cell_4/BiasAdd_3:output:0)lstm/while/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/while/lstm_cell_4/add_4Ѓ
 lstm/while/lstm_cell_4/Sigmoid_2Sigmoid lstm/while/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2"
 lstm/while/lstm_cell_4/Sigmoid_2
lstm/while/lstm_cell_4/Tanh_1Tanh lstm/while/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/while/lstm_cell_4/Tanh_1Р
lstm/while/lstm_cell_4/mul_10Mul$lstm/while/lstm_cell_4/Sigmoid_2:y:0!lstm/while/lstm_cell_4/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/while/lstm_cell_4/mul_10љ
/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_while_placeholder_1lstm_while_placeholder!lstm/while/lstm_cell_4/mul_10:z:0*
_output_shapes
: *
element_dtype021
/lstm/while/TensorArrayV2Write/TensorListSetItemf
lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/add/y}
lstm/while/addAddV2lstm_while_placeholderlstm/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm/while/addj
lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/add_1/y
lstm/while/add_1AddV2"lstm_while_lstm_while_loop_counterlstm/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm/while/add_1m
lstm/while/IdentityIdentitylstm/while/add_1:z:0*
T0*
_output_shapes
: 2
lstm/while/Identity
lstm/while/Identity_1Identity(lstm_while_lstm_while_maximum_iterations*
T0*
_output_shapes
: 2
lstm/while/Identity_1o
lstm/while/Identity_2Identitylstm/while/add:z:0*
T0*
_output_shapes
: 2
lstm/while/Identity_2
lstm/while/Identity_3Identity?lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
lstm/while/Identity_3
lstm/while/Identity_4Identity!lstm/while/lstm_cell_4/mul_10:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/while/Identity_4
lstm/while/Identity_5Identity lstm/while/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/while/Identity_5"3
lstm_while_identitylstm/while/Identity:output:0"7
lstm_while_identity_1lstm/while/Identity_1:output:0"7
lstm_while_identity_2lstm/while/Identity_2:output:0"7
lstm_while_identity_3lstm/while/Identity_3:output:0"7
lstm_while_identity_4lstm/while/Identity_4:output:0"7
lstm_while_identity_5lstm/while/Identity_5:output:0"b
.lstm_while_lstm_cell_4_readvariableop_resource0lstm_while_lstm_cell_4_readvariableop_resource_0"r
6lstm_while_lstm_cell_4_split_1_readvariableop_resource8lstm_while_lstm_cell_4_split_1_readvariableop_resource_0"n
4lstm_while_lstm_cell_4_split_readvariableop_resource6lstm_while_lstm_cell_4_split_readvariableop_resource_0"D
lstm_while_lstm_strided_slice_1!lstm_while_lstm_strided_slice_1_0"М
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџd:џџџџџџџџџd: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
: 
І
О
while_cond_31925
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_31925___redundant_placeholder03
/while_while_cond_31925___redundant_placeholder13
/while_while_cond_31925___redundant_placeholder23
/while_while_cond_31925___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџd:џџџџџџџџџd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
:


$__inference_lstm_layer_call_fn_34232
inputs_0
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_318632
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:џџџџџџџџџџџџџџџџџџЌ:::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ
"
_user_specified_name
inputs/0
ќ4
Я	
__inference__traced_save_35293
file_prefix3
/savev2_embedding_embeddings_read_readvariableop1
-savev2_dense_final_kernel_read_readvariableop/
+savev2_dense_final_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop6
2savev2_lstm_lstm_cell_4_kernel_read_readvariableop@
<savev2_lstm_lstm_cell_4_recurrent_kernel_read_readvariableop4
0savev2_lstm_lstm_cell_4_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop?
;savev2_rmsprop_embedding_embeddings_rms_read_readvariableop=
9savev2_rmsprop_dense_final_kernel_rms_read_readvariableop;
7savev2_rmsprop_dense_final_bias_rms_read_readvariableopB
>savev2_rmsprop_lstm_lstm_cell_4_kernel_rms_read_readvariableopL
Hsavev2_rmsprop_lstm_lstm_cell_4_recurrent_kernel_rms_read_readvariableop@
<savev2_rmsprop_lstm_lstm_cell_4_bias_rms_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_60e8ff1cdb3043abadb1d1b4ca1443bb/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ѓ

value
B
B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesД
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesп	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_embedding_embeddings_read_readvariableop-savev2_dense_final_kernel_read_readvariableop+savev2_dense_final_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop2savev2_lstm_lstm_cell_4_kernel_read_readvariableop<savev2_lstm_lstm_cell_4_recurrent_kernel_read_readvariableop0savev2_lstm_lstm_cell_4_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop;savev2_rmsprop_embedding_embeddings_rms_read_readvariableop9savev2_rmsprop_dense_final_kernel_rms_read_readvariableop7savev2_rmsprop_dense_final_bias_rms_read_readvariableop>savev2_rmsprop_lstm_lstm_cell_4_kernel_rms_read_readvariableopHsavev2_rmsprop_lstm_lstm_cell_4_recurrent_kernel_rms_read_readvariableop<savev2_rmsprop_lstm_lstm_cell_4_bias_rms_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *$
dtypes
2	2
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*Ё
_input_shapes
: :ЌЌ:d:: : : : : :
Ќ:	d:: : : : :ЌЌ:d::
Ќ:	d:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:'#
!
_output_shapes
:ЌЌ:$ 

_output_shapes

:d: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&	"
 
_output_shapes
:
Ќ:%
!

_output_shapes
:	d:!

_output_shapes	
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :'#
!
_output_shapes
:ЌЌ:$ 

_output_shapes

:d: 

_output_shapes
::&"
 
_output_shapes
:
Ќ:%!

_output_shapes
:	d:!

_output_shapes	
::

_output_shapes
: 
х
п
while_body_32212
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_05
1while_lstm_cell_4_split_readvariableop_resource_07
3while_lstm_cell_4_split_1_readvariableop_resource_0/
+while_lstm_cell_4_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor3
/while_lstm_cell_4_split_readvariableop_resource5
1while_lstm_cell_4_split_1_readvariableop_resource-
)while_lstm_cell_4_readvariableop_resourceУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeд
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџЌ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemІ
!while/lstm_cell_4/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/ones_like/Shape
!while/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!while/lstm_cell_4/ones_like/ConstЭ
while/lstm_cell_4/ones_likeFill*while/lstm_cell_4/ones_like/Shape:output:0*while/lstm_cell_4/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_4/ones_like
while/lstm_cell_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2!
while/lstm_cell_4/dropout/ConstШ
while/lstm_cell_4/dropout/MulMul$while/lstm_cell_4/ones_like:output:0(while/lstm_cell_4/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_4/dropout/Mul
while/lstm_cell_4/dropout/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell_4/dropout/Shape
6while/lstm_cell_4/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_4/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*
seedБџх)*
seed2БЄ28
6while/lstm_cell_4/dropout/random_uniform/RandomUniform
(while/lstm_cell_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2*
(while/lstm_cell_4/dropout/GreaterEqual/y
&while/lstm_cell_4/dropout/GreaterEqualGreaterEqual?while/lstm_cell_4/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_4/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2(
&while/lstm_cell_4/dropout/GreaterEqualЖ
while/lstm_cell_4/dropout/CastCast*while/lstm_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2 
while/lstm_cell_4/dropout/CastУ
while/lstm_cell_4/dropout/Mul_1Mul!while/lstm_cell_4/dropout/Mul:z:0"while/lstm_cell_4/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2!
while/lstm_cell_4/dropout/Mul_1
!while/lstm_cell_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2#
!while/lstm_cell_4/dropout_1/ConstЮ
while/lstm_cell_4/dropout_1/MulMul$while/lstm_cell_4/ones_like:output:0*while/lstm_cell_4/dropout_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2!
while/lstm_cell_4/dropout_1/Mul
!while/lstm_cell_4/dropout_1/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/dropout_1/Shape
8while/lstm_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_1/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*
seedБџх)*
seed2рF2:
8while/lstm_cell_4/dropout_1/random_uniform/RandomUniform
*while/lstm_cell_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2,
*while/lstm_cell_4/dropout_1/GreaterEqual/y
(while/lstm_cell_4/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2*
(while/lstm_cell_4/dropout_1/GreaterEqualМ
 while/lstm_cell_4/dropout_1/CastCast,while/lstm_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2"
 while/lstm_cell_4/dropout_1/CastЫ
!while/lstm_cell_4/dropout_1/Mul_1Mul#while/lstm_cell_4/dropout_1/Mul:z:0$while/lstm_cell_4/dropout_1/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2#
!while/lstm_cell_4/dropout_1/Mul_1
!while/lstm_cell_4/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2#
!while/lstm_cell_4/dropout_2/ConstЮ
while/lstm_cell_4/dropout_2/MulMul$while/lstm_cell_4/ones_like:output:0*while/lstm_cell_4/dropout_2/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2!
while/lstm_cell_4/dropout_2/Mul
!while/lstm_cell_4/dropout_2/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/dropout_2/Shape
8while/lstm_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_2/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*
seedБџх)*
seed2к2:
8while/lstm_cell_4/dropout_2/random_uniform/RandomUniform
*while/lstm_cell_4/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2,
*while/lstm_cell_4/dropout_2/GreaterEqual/y
(while/lstm_cell_4/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2*
(while/lstm_cell_4/dropout_2/GreaterEqualМ
 while/lstm_cell_4/dropout_2/CastCast,while/lstm_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2"
 while/lstm_cell_4/dropout_2/CastЫ
!while/lstm_cell_4/dropout_2/Mul_1Mul#while/lstm_cell_4/dropout_2/Mul:z:0$while/lstm_cell_4/dropout_2/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2#
!while/lstm_cell_4/dropout_2/Mul_1
!while/lstm_cell_4/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2#
!while/lstm_cell_4/dropout_3/ConstЮ
while/lstm_cell_4/dropout_3/MulMul$while/lstm_cell_4/ones_like:output:0*while/lstm_cell_4/dropout_3/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2!
while/lstm_cell_4/dropout_3/Mul
!while/lstm_cell_4/dropout_3/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/dropout_3/Shape
8while/lstm_cell_4/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_3/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*
seedБџх)*
seed2Чб2:
8while/lstm_cell_4/dropout_3/random_uniform/RandomUniform
*while/lstm_cell_4/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2,
*while/lstm_cell_4/dropout_3/GreaterEqual/y
(while/lstm_cell_4/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2*
(while/lstm_cell_4/dropout_3/GreaterEqualМ
 while/lstm_cell_4/dropout_3/CastCast,while/lstm_cell_4/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2"
 while/lstm_cell_4/dropout_3/CastЫ
!while/lstm_cell_4/dropout_3/Mul_1Mul#while/lstm_cell_4/dropout_3/Mul:z:0$while/lstm_cell_4/dropout_3/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2#
!while/lstm_cell_4/dropout_3/Mul_1
#while/lstm_cell_4/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2%
#while/lstm_cell_4/ones_like_1/Shape
#while/lstm_cell_4/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2%
#while/lstm_cell_4/ones_like_1/Constд
while/lstm_cell_4/ones_like_1Fill,while/lstm_cell_4/ones_like_1/Shape:output:0,while/lstm_cell_4/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/ones_like_1
!while/lstm_cell_4/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2#
!while/lstm_cell_4/dropout_4/ConstЯ
while/lstm_cell_4/dropout_4/MulMul&while/lstm_cell_4/ones_like_1:output:0*while/lstm_cell_4/dropout_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2!
while/lstm_cell_4/dropout_4/Mul
!while/lstm_cell_4/dropout_4/ShapeShape&while/lstm_cell_4/ones_like_1:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/dropout_4/Shape
8while/lstm_cell_4/dropout_4/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџd*
dtype0*
seedБџх)*
seed2Іѕу2:
8while/lstm_cell_4/dropout_4/random_uniform/RandomUniform
*while/lstm_cell_4/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2,
*while/lstm_cell_4/dropout_4/GreaterEqual/y
(while/lstm_cell_4/dropout_4/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_4/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2*
(while/lstm_cell_4/dropout_4/GreaterEqualЛ
 while/lstm_cell_4/dropout_4/CastCast,while/lstm_cell_4/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd2"
 while/lstm_cell_4/dropout_4/CastЪ
!while/lstm_cell_4/dropout_4/Mul_1Mul#while/lstm_cell_4/dropout_4/Mul:z:0$while/lstm_cell_4/dropout_4/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2#
!while/lstm_cell_4/dropout_4/Mul_1
!while/lstm_cell_4/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2#
!while/lstm_cell_4/dropout_5/ConstЯ
while/lstm_cell_4/dropout_5/MulMul&while/lstm_cell_4/ones_like_1:output:0*while/lstm_cell_4/dropout_5/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2!
while/lstm_cell_4/dropout_5/Mul
!while/lstm_cell_4/dropout_5/ShapeShape&while/lstm_cell_4/ones_like_1:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/dropout_5/Shape
8while/lstm_cell_4/dropout_5/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџd*
dtype0*
seedБџх)*
seed2В2:
8while/lstm_cell_4/dropout_5/random_uniform/RandomUniform
*while/lstm_cell_4/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2,
*while/lstm_cell_4/dropout_5/GreaterEqual/y
(while/lstm_cell_4/dropout_5/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_5/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2*
(while/lstm_cell_4/dropout_5/GreaterEqualЛ
 while/lstm_cell_4/dropout_5/CastCast,while/lstm_cell_4/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd2"
 while/lstm_cell_4/dropout_5/CastЪ
!while/lstm_cell_4/dropout_5/Mul_1Mul#while/lstm_cell_4/dropout_5/Mul:z:0$while/lstm_cell_4/dropout_5/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2#
!while/lstm_cell_4/dropout_5/Mul_1
!while/lstm_cell_4/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2#
!while/lstm_cell_4/dropout_6/ConstЯ
while/lstm_cell_4/dropout_6/MulMul&while/lstm_cell_4/ones_like_1:output:0*while/lstm_cell_4/dropout_6/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2!
while/lstm_cell_4/dropout_6/Mul
!while/lstm_cell_4/dropout_6/ShapeShape&while/lstm_cell_4/ones_like_1:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/dropout_6/Shape
8while/lstm_cell_4/dropout_6/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_6/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџd*
dtype0*
seedБџх)*
seed2ў2:
8while/lstm_cell_4/dropout_6/random_uniform/RandomUniform
*while/lstm_cell_4/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2,
*while/lstm_cell_4/dropout_6/GreaterEqual/y
(while/lstm_cell_4/dropout_6/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_6/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2*
(while/lstm_cell_4/dropout_6/GreaterEqualЛ
 while/lstm_cell_4/dropout_6/CastCast,while/lstm_cell_4/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd2"
 while/lstm_cell_4/dropout_6/CastЪ
!while/lstm_cell_4/dropout_6/Mul_1Mul#while/lstm_cell_4/dropout_6/Mul:z:0$while/lstm_cell_4/dropout_6/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2#
!while/lstm_cell_4/dropout_6/Mul_1
!while/lstm_cell_4/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2#
!while/lstm_cell_4/dropout_7/ConstЯ
while/lstm_cell_4/dropout_7/MulMul&while/lstm_cell_4/ones_like_1:output:0*while/lstm_cell_4/dropout_7/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2!
while/lstm_cell_4/dropout_7/Mul
!while/lstm_cell_4/dropout_7/ShapeShape&while/lstm_cell_4/ones_like_1:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/dropout_7/Shape
8while/lstm_cell_4/dropout_7/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_7/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџd*
dtype0*
seedБџх)*
seed2Ђ№2:
8while/lstm_cell_4/dropout_7/random_uniform/RandomUniform
*while/lstm_cell_4/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2,
*while/lstm_cell_4/dropout_7/GreaterEqual/y
(while/lstm_cell_4/dropout_7/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_7/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2*
(while/lstm_cell_4/dropout_7/GreaterEqualЛ
 while/lstm_cell_4/dropout_7/CastCast,while/lstm_cell_4/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd2"
 while/lstm_cell_4/dropout_7/CastЪ
!while/lstm_cell_4/dropout_7/Mul_1Mul#while/lstm_cell_4/dropout_7/Mul:z:0$while/lstm_cell_4/dropout_7/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2#
!while/lstm_cell_4/dropout_7/Mul_1П
while/lstm_cell_4/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/lstm_cell_4/dropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_4/mulХ
while/lstm_cell_4/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_4/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_4/mul_1Х
while/lstm_cell_4/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_4/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_4/mul_2Х
while/lstm_cell_4/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_4/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_4/mul_3t
while/lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_4/Const
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_4/split/split_dimФ
&while/lstm_cell_4/split/ReadVariableOpReadVariableOp1while_lstm_cell_4_split_readvariableop_resource_0* 
_output_shapes
:
Ќ*
dtype02(
&while/lstm_cell_4/split/ReadVariableOpѓ
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0.while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	Ќd:	Ќd:	Ќd:	Ќd*
	num_split2
while/lstm_cell_4/split­
while/lstm_cell_4/MatMulMatMulwhile/lstm_cell_4/mul:z:0 while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMulГ
while/lstm_cell_4/MatMul_1MatMulwhile/lstm_cell_4/mul_1:z:0 while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMul_1Г
while/lstm_cell_4/MatMul_2MatMulwhile/lstm_cell_4/mul_2:z:0 while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMul_2Г
while/lstm_cell_4/MatMul_3MatMulwhile/lstm_cell_4/mul_3:z:0 while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMul_3x
while/lstm_cell_4/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_4/Const_1
#while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_4/split_1/split_dimХ
(while/lstm_cell_4/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02*
(while/lstm_cell_4/split_1/ReadVariableOpч
while/lstm_cell_4/split_1Split,while/lstm_cell_4/split_1/split_dim:output:00while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:d:d:d:d*
	num_split2
while/lstm_cell_4/split_1Л
while/lstm_cell_4/BiasAddBiasAdd"while/lstm_cell_4/MatMul:product:0"while/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/BiasAddС
while/lstm_cell_4/BiasAdd_1BiasAdd$while/lstm_cell_4/MatMul_1:product:0"while/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/BiasAdd_1С
while/lstm_cell_4/BiasAdd_2BiasAdd$while/lstm_cell_4/MatMul_2:product:0"while/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/BiasAdd_2С
while/lstm_cell_4/BiasAdd_3BiasAdd$while/lstm_cell_4/MatMul_3:product:0"while/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/BiasAdd_3Ї
while/lstm_cell_4/mul_4Mulwhile_placeholder_2%while/lstm_cell_4/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/mul_4Ї
while/lstm_cell_4/mul_5Mulwhile_placeholder_2%while/lstm_cell_4/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/mul_5Ї
while/lstm_cell_4/mul_6Mulwhile_placeholder_2%while/lstm_cell_4/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/mul_6Ї
while/lstm_cell_4/mul_7Mulwhile_placeholder_2%while/lstm_cell_4/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/mul_7Б
 while/lstm_cell_4/ReadVariableOpReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	d*
dtype02"
 while/lstm_cell_4/ReadVariableOp
%while/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell_4/strided_slice/stackЃ
'while/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   2)
'while/lstm_cell_4/strided_slice/stack_1Ѓ
'while/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell_4/strided_slice/stack_2ш
while/lstm_cell_4/strided_sliceStridedSlice(while/lstm_cell_4/ReadVariableOp:value:0.while/lstm_cell_4/strided_slice/stack:output:00while/lstm_cell_4/strided_slice/stack_1:output:00while/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2!
while/lstm_cell_4/strided_sliceЛ
while/lstm_cell_4/MatMul_4MatMulwhile/lstm_cell_4/mul_4:z:0(while/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMul_4Г
while/lstm_cell_4/addAddV2"while/lstm_cell_4/BiasAdd:output:0$while/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/add
while/lstm_cell_4/SigmoidSigmoidwhile/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/SigmoidЕ
"while/lstm_cell_4/ReadVariableOp_1ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	d*
dtype02$
"while/lstm_cell_4/ReadVariableOp_1Ѓ
'while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   2)
'while/lstm_cell_4/strided_slice_1/stackЇ
)while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Ш   2+
)while/lstm_cell_4/strided_slice_1/stack_1Ї
)while/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_4/strided_slice_1/stack_2є
!while/lstm_cell_4/strided_slice_1StridedSlice*while/lstm_cell_4/ReadVariableOp_1:value:00while/lstm_cell_4/strided_slice_1/stack:output:02while/lstm_cell_4/strided_slice_1/stack_1:output:02while/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_1Н
while/lstm_cell_4/MatMul_5MatMulwhile/lstm_cell_4/mul_5:z:0*while/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMul_5Й
while/lstm_cell_4/add_1AddV2$while/lstm_cell_4/BiasAdd_1:output:0$while/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/add_1
while/lstm_cell_4/Sigmoid_1Sigmoidwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/Sigmoid_1Ё
while/lstm_cell_4/mul_8Mulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/mul_8Е
"while/lstm_cell_4/ReadVariableOp_2ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	d*
dtype02$
"while/lstm_cell_4/ReadVariableOp_2Ѓ
'while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    Ш   2)
'while/lstm_cell_4/strided_slice_2/stackЇ
)while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  2+
)while/lstm_cell_4/strided_slice_2/stack_1Ї
)while/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_4/strided_slice_2/stack_2є
!while/lstm_cell_4/strided_slice_2StridedSlice*while/lstm_cell_4/ReadVariableOp_2:value:00while/lstm_cell_4/strided_slice_2/stack:output:02while/lstm_cell_4/strided_slice_2/stack_1:output:02while/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_2Н
while/lstm_cell_4/MatMul_6MatMulwhile/lstm_cell_4/mul_6:z:0*while/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMul_6Й
while/lstm_cell_4/add_2AddV2$while/lstm_cell_4/BiasAdd_2:output:0$while/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/add_2
while/lstm_cell_4/TanhTanhwhile/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/TanhІ
while/lstm_cell_4/mul_9Mulwhile/lstm_cell_4/Sigmoid:y:0while/lstm_cell_4/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/mul_9Ї
while/lstm_cell_4/add_3AddV2while/lstm_cell_4/mul_8:z:0while/lstm_cell_4/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/add_3Е
"while/lstm_cell_4/ReadVariableOp_3ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	d*
dtype02$
"while/lstm_cell_4/ReadVariableOp_3Ѓ
'while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  2)
'while/lstm_cell_4/strided_slice_3/stackЇ
)while/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_4/strided_slice_3/stack_1Ї
)while/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_4/strided_slice_3/stack_2є
!while/lstm_cell_4/strided_slice_3StridedSlice*while/lstm_cell_4/ReadVariableOp_3:value:00while/lstm_cell_4/strided_slice_3/stack:output:02while/lstm_cell_4/strided_slice_3/stack_1:output:02while/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_3Н
while/lstm_cell_4/MatMul_7MatMulwhile/lstm_cell_4/mul_7:z:0*while/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMul_7Й
while/lstm_cell_4/add_4AddV2$while/lstm_cell_4/BiasAdd_3:output:0$while/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/add_4
while/lstm_cell_4/Sigmoid_2Sigmoidwhile/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/Sigmoid_2
while/lstm_cell_4/Tanh_1Tanhwhile/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/Tanh_1Ќ
while/lstm_cell_4/mul_10Mulwhile/lstm_cell_4/Sigmoid_2:y:0while/lstm_cell_4/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/mul_10р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_4/mul_10:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_4/mul_10:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_4_readvariableop_resource+while_lstm_cell_4_readvariableop_resource_0"h
1while_lstm_cell_4_split_1_readvariableop_resource3while_lstm_cell_4_split_1_readvariableop_resource_0"d
/while_lstm_cell_4_split_readvariableop_resource1while_lstm_cell_4_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџd:џџџџџџџџџd: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
: 
ПD
б
?__inference_lstm_layer_call_and_return_conditional_losses_31863

inputs
lstm_cell_4_31781
lstm_cell_4_31783
lstm_cell_4_31785
identityЂ#lstm_cell_4/StatefulPartitionedCallЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2§
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
shrink_axis_mask2
strided_slice_2
#lstm_cell_4/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_4_31781lstm_cell_4_31783lstm_cell_4_31785*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_314162%
#lstm_cell_4/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_4_31781lstm_cell_4_31783lstm_cell_4_31785*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_31794*
condR
while_cond_31793*K
output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџd*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџd2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_4/StatefulPartitionedCall^while*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:џџџџџџџџџџџџџџџџџџЌ:::2J
#lstm_cell_4/StatefulPartitionedCall#lstm_cell_4/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ
 
_user_specified_nameinputs
І
О
while_cond_34425
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_34425___redundant_placeholder03
/while_while_cond_34425___redundant_placeholder13
/while_while_cond_34425___redundant_placeholder23
/while_while_cond_34425___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџd:џџџџџџџџџd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
:
І
О
while_cond_32211
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_32211___redundant_placeholder03
/while_while_cond_32211___redundant_placeholder13
/while_while_cond_32211___redundant_placeholder23
/while_while_cond_32211___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџd:џџџџџџџџџd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
:
І
О
while_cond_34744
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_34744___redundant_placeholder03
/while_while_cond_34744___redundant_placeholder13
/while_while_cond_34744___redundant_placeholder23
/while_while_cond_34744___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџd:џџџџџџџџџd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
:
я

F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_31416

inputs

states
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2X
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like/Const
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2
dropout/Const
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/Shapeд
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*
seedБџх)*
seed2ЮЦТ2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2
dropout/GreaterEqual/yП
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2
dropout_1/Const
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/Shapeк
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*
seedБџх)*
seed2зЅ2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2
dropout_1/GreaterEqual/yЧ
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
dropout_1/GreaterEqual
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
dropout_1/Cast
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2
dropout_2/Const
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/Shapeк
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*
seedБџх)*
seed2к2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2
dropout_2/GreaterEqual/yЧ
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
dropout_2/GreaterEqual
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
dropout_2/Cast
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
dropout_2/Mul_1g
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2
dropout_3/Const
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
dropout_3/Muld
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_3/Shapeк
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*
seedБџх)*
seed2УжЪ2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2
dropout_3/GreaterEqual/yЧ
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
dropout_3/GreaterEqual
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
dropout_3/Cast
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
dropout_3/Mul_1\
ones_like_1/ShapeShapestates*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like_1/Const
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
ones_like_1g
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2
dropout_4/Const
dropout_4/MulMulones_like_1:output:0dropout_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
dropout_4/Mulf
dropout_4/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_4/Shapeй
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџd*
dtype0*
seedБџх)*
seed2зі2(
&dropout_4/random_uniform/RandomUniformy
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2
dropout_4/GreaterEqual/yЦ
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
dropout_4/GreaterEqual
dropout_4/CastCastdropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd2
dropout_4/Cast
dropout_4/Mul_1Muldropout_4/Mul:z:0dropout_4/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
dropout_4/Mul_1g
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2
dropout_5/Const
dropout_5/MulMulones_like_1:output:0dropout_5/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
dropout_5/Mulf
dropout_5/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_5/Shapeй
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџd*
dtype0*
seedБџх)*
seed2щы2(
&dropout_5/random_uniform/RandomUniformy
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2
dropout_5/GreaterEqual/yЦ
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
dropout_5/GreaterEqual
dropout_5/CastCastdropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd2
dropout_5/Cast
dropout_5/Mul_1Muldropout_5/Mul:z:0dropout_5/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
dropout_5/Mul_1g
dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2
dropout_6/Const
dropout_6/MulMulones_like_1:output:0dropout_6/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
dropout_6/Mulf
dropout_6/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_6/Shapeй
&dropout_6/random_uniform/RandomUniformRandomUniformdropout_6/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџd*
dtype0*
seedБџх)*
seed2лЯ2(
&dropout_6/random_uniform/RandomUniformy
dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2
dropout_6/GreaterEqual/yЦ
dropout_6/GreaterEqualGreaterEqual/dropout_6/random_uniform/RandomUniform:output:0!dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
dropout_6/GreaterEqual
dropout_6/CastCastdropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd2
dropout_6/Cast
dropout_6/Mul_1Muldropout_6/Mul:z:0dropout_6/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
dropout_6/Mul_1g
dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2
dropout_7/Const
dropout_7/MulMulones_like_1:output:0dropout_7/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
dropout_7/Mulf
dropout_7/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_7/Shapeи
&dropout_7/random_uniform/RandomUniformRandomUniformdropout_7/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџd*
dtype0*
seedБџх)*
seed2ЬС*2(
&dropout_7/random_uniform/RandomUniformy
dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2
dropout_7/GreaterEqual/yЦ
dropout_7/GreaterEqualGreaterEqual/dropout_7/random_uniform/RandomUniform:output:0!dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
dropout_7/GreaterEqual
dropout_7/CastCastdropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd2
dropout_7/Cast
dropout_7/Mul_1Muldropout_7/Mul:z:0dropout_7/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
dropout_7/Mul_1_
mulMulinputsdropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
mule
mul_1Mulinputsdropout_1/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
mul_1e
mul_2Mulinputsdropout_2/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
mul_2e
mul_3Mulinputsdropout_3/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
mul_3P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
Ќ*
dtype02
split/ReadVariableOpЋ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	Ќd:	Ќd:	Ќd:	Ќd*
	num_split2
splite
MatMulMatMulmul:z:0split:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
MatMulk
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd2

MatMul_1k
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd2

MatMul_2k
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*'
_output_shapes
:џџџџџџџџџd2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype02
split_1/ReadVariableOp
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:d:d:d:d*
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџd2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџd2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџd2
	BiasAdd_3d
mul_4Mulstatesdropout_4/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
mul_4d
mul_5Mulstatesdropout_5/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
mul_5d
mul_6Mulstatesdropout_6/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
mul_6d
mul_7Mulstatesdropout_7/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
mul_7y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	d*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2ќ
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2
strided_slices
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2	
Sigmoid}
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	d*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Ш   2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2
strided_slice_1u
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2

MatMul_5q
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
	Sigmoid_1`
mul_8MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџd2
mul_8}
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	d*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    Ш   2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2
strided_slice_2u
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2

MatMul_6q
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
Tanh^
mul_9MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
mul_9_
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
add_3}
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	d*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  2
strided_slice_3/stack
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2
strided_slice_3u
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2

MatMul_7q
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
	Sigmoid_2U
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
Tanh_1d
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
mul_10^
IdentityIdentity
mul_10:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2

Identityb

Identity_1Identity
mul_10:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity_1a

Identity_2Identity	add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:џџџџџџџџџЌ:џџџџџџџџџd:џџџџџџџџџd::::P L
(
_output_shapes
:џџџџџџџџџЌ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_namestates:OK
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_namestates
Г
Ў
F__inference_dense_final_layer_call_and_return_conditional_losses_32729

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџd:::O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
В

D__inference_embedding_layer_call_and_return_conditional_losses_32018

inputs
embedding_lookup_32012
identity]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ(2
CastЮ
embedding_lookupResourceGatherembedding_lookup_32012Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/32012*,
_output_shapes
:џџџџџџџџџ(Ќ*
dtype02
embedding_lookupП
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/32012*,
_output_shapes
:џџџџџџџџџ(Ќ2
embedding_lookup/IdentityЁ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:џџџџџџџџџ(Ќ2
embedding_lookup/Identity_1}
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ(Ќ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ(::O K
'
_output_shapes
:џџџџџџџџџ(
 
_user_specified_nameinputs
х
п
while_body_33766
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_05
1while_lstm_cell_4_split_readvariableop_resource_07
3while_lstm_cell_4_split_1_readvariableop_resource_0/
+while_lstm_cell_4_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor3
/while_lstm_cell_4_split_readvariableop_resource5
1while_lstm_cell_4_split_1_readvariableop_resource-
)while_lstm_cell_4_readvariableop_resourceУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeд
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџЌ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemІ
!while/lstm_cell_4/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/ones_like/Shape
!while/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!while/lstm_cell_4/ones_like/ConstЭ
while/lstm_cell_4/ones_likeFill*while/lstm_cell_4/ones_like/Shape:output:0*while/lstm_cell_4/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_4/ones_like
while/lstm_cell_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2!
while/lstm_cell_4/dropout/ConstШ
while/lstm_cell_4/dropout/MulMul$while/lstm_cell_4/ones_like:output:0(while/lstm_cell_4/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_4/dropout/Mul
while/lstm_cell_4/dropout/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell_4/dropout/Shape
6while/lstm_cell_4/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_4/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*
seedБџх)*
seed2Рp28
6while/lstm_cell_4/dropout/random_uniform/RandomUniform
(while/lstm_cell_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2*
(while/lstm_cell_4/dropout/GreaterEqual/y
&while/lstm_cell_4/dropout/GreaterEqualGreaterEqual?while/lstm_cell_4/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_4/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2(
&while/lstm_cell_4/dropout/GreaterEqualЖ
while/lstm_cell_4/dropout/CastCast*while/lstm_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2 
while/lstm_cell_4/dropout/CastУ
while/lstm_cell_4/dropout/Mul_1Mul!while/lstm_cell_4/dropout/Mul:z:0"while/lstm_cell_4/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2!
while/lstm_cell_4/dropout/Mul_1
!while/lstm_cell_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2#
!while/lstm_cell_4/dropout_1/ConstЮ
while/lstm_cell_4/dropout_1/MulMul$while/lstm_cell_4/ones_like:output:0*while/lstm_cell_4/dropout_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2!
while/lstm_cell_4/dropout_1/Mul
!while/lstm_cell_4/dropout_1/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/dropout_1/Shape
8while/lstm_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_1/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*
seedБџх)*
seed2ыU2:
8while/lstm_cell_4/dropout_1/random_uniform/RandomUniform
*while/lstm_cell_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2,
*while/lstm_cell_4/dropout_1/GreaterEqual/y
(while/lstm_cell_4/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2*
(while/lstm_cell_4/dropout_1/GreaterEqualМ
 while/lstm_cell_4/dropout_1/CastCast,while/lstm_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2"
 while/lstm_cell_4/dropout_1/CastЫ
!while/lstm_cell_4/dropout_1/Mul_1Mul#while/lstm_cell_4/dropout_1/Mul:z:0$while/lstm_cell_4/dropout_1/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2#
!while/lstm_cell_4/dropout_1/Mul_1
!while/lstm_cell_4/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2#
!while/lstm_cell_4/dropout_2/ConstЮ
while/lstm_cell_4/dropout_2/MulMul$while/lstm_cell_4/ones_like:output:0*while/lstm_cell_4/dropout_2/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2!
while/lstm_cell_4/dropout_2/Mul
!while/lstm_cell_4/dropout_2/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/dropout_2/Shape
8while/lstm_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_2/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*
seedБџх)*
seed2юО2:
8while/lstm_cell_4/dropout_2/random_uniform/RandomUniform
*while/lstm_cell_4/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2,
*while/lstm_cell_4/dropout_2/GreaterEqual/y
(while/lstm_cell_4/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2*
(while/lstm_cell_4/dropout_2/GreaterEqualМ
 while/lstm_cell_4/dropout_2/CastCast,while/lstm_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2"
 while/lstm_cell_4/dropout_2/CastЫ
!while/lstm_cell_4/dropout_2/Mul_1Mul#while/lstm_cell_4/dropout_2/Mul:z:0$while/lstm_cell_4/dropout_2/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2#
!while/lstm_cell_4/dropout_2/Mul_1
!while/lstm_cell_4/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2#
!while/lstm_cell_4/dropout_3/ConstЮ
while/lstm_cell_4/dropout_3/MulMul$while/lstm_cell_4/ones_like:output:0*while/lstm_cell_4/dropout_3/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2!
while/lstm_cell_4/dropout_3/Mul
!while/lstm_cell_4/dropout_3/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/dropout_3/Shape
8while/lstm_cell_4/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_3/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*
seedБџх)*
seed2вј2:
8while/lstm_cell_4/dropout_3/random_uniform/RandomUniform
*while/lstm_cell_4/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2,
*while/lstm_cell_4/dropout_3/GreaterEqual/y
(while/lstm_cell_4/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2*
(while/lstm_cell_4/dropout_3/GreaterEqualМ
 while/lstm_cell_4/dropout_3/CastCast,while/lstm_cell_4/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2"
 while/lstm_cell_4/dropout_3/CastЫ
!while/lstm_cell_4/dropout_3/Mul_1Mul#while/lstm_cell_4/dropout_3/Mul:z:0$while/lstm_cell_4/dropout_3/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2#
!while/lstm_cell_4/dropout_3/Mul_1
#while/lstm_cell_4/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2%
#while/lstm_cell_4/ones_like_1/Shape
#while/lstm_cell_4/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2%
#while/lstm_cell_4/ones_like_1/Constд
while/lstm_cell_4/ones_like_1Fill,while/lstm_cell_4/ones_like_1/Shape:output:0,while/lstm_cell_4/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/ones_like_1
!while/lstm_cell_4/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2#
!while/lstm_cell_4/dropout_4/ConstЯ
while/lstm_cell_4/dropout_4/MulMul&while/lstm_cell_4/ones_like_1:output:0*while/lstm_cell_4/dropout_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2!
while/lstm_cell_4/dropout_4/Mul
!while/lstm_cell_4/dropout_4/ShapeShape&while/lstm_cell_4/ones_like_1:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/dropout_4/Shape
8while/lstm_cell_4/dropout_4/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџd*
dtype0*
seedБџх)*
seed2ќ2:
8while/lstm_cell_4/dropout_4/random_uniform/RandomUniform
*while/lstm_cell_4/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2,
*while/lstm_cell_4/dropout_4/GreaterEqual/y
(while/lstm_cell_4/dropout_4/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_4/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2*
(while/lstm_cell_4/dropout_4/GreaterEqualЛ
 while/lstm_cell_4/dropout_4/CastCast,while/lstm_cell_4/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd2"
 while/lstm_cell_4/dropout_4/CastЪ
!while/lstm_cell_4/dropout_4/Mul_1Mul#while/lstm_cell_4/dropout_4/Mul:z:0$while/lstm_cell_4/dropout_4/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2#
!while/lstm_cell_4/dropout_4/Mul_1
!while/lstm_cell_4/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2#
!while/lstm_cell_4/dropout_5/ConstЯ
while/lstm_cell_4/dropout_5/MulMul&while/lstm_cell_4/ones_like_1:output:0*while/lstm_cell_4/dropout_5/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2!
while/lstm_cell_4/dropout_5/Mul
!while/lstm_cell_4/dropout_5/ShapeShape&while/lstm_cell_4/ones_like_1:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/dropout_5/Shape
8while/lstm_cell_4/dropout_5/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџd*
dtype0*
seedБџх)*
seed2тьб2:
8while/lstm_cell_4/dropout_5/random_uniform/RandomUniform
*while/lstm_cell_4/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2,
*while/lstm_cell_4/dropout_5/GreaterEqual/y
(while/lstm_cell_4/dropout_5/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_5/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2*
(while/lstm_cell_4/dropout_5/GreaterEqualЛ
 while/lstm_cell_4/dropout_5/CastCast,while/lstm_cell_4/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd2"
 while/lstm_cell_4/dropout_5/CastЪ
!while/lstm_cell_4/dropout_5/Mul_1Mul#while/lstm_cell_4/dropout_5/Mul:z:0$while/lstm_cell_4/dropout_5/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2#
!while/lstm_cell_4/dropout_5/Mul_1
!while/lstm_cell_4/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2#
!while/lstm_cell_4/dropout_6/ConstЯ
while/lstm_cell_4/dropout_6/MulMul&while/lstm_cell_4/ones_like_1:output:0*while/lstm_cell_4/dropout_6/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2!
while/lstm_cell_4/dropout_6/Mul
!while/lstm_cell_4/dropout_6/ShapeShape&while/lstm_cell_4/ones_like_1:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/dropout_6/Shape
8while/lstm_cell_4/dropout_6/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_6/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџd*
dtype0*
seedБџх)*
seed2аЊј2:
8while/lstm_cell_4/dropout_6/random_uniform/RandomUniform
*while/lstm_cell_4/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2,
*while/lstm_cell_4/dropout_6/GreaterEqual/y
(while/lstm_cell_4/dropout_6/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_6/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2*
(while/lstm_cell_4/dropout_6/GreaterEqualЛ
 while/lstm_cell_4/dropout_6/CastCast,while/lstm_cell_4/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd2"
 while/lstm_cell_4/dropout_6/CastЪ
!while/lstm_cell_4/dropout_6/Mul_1Mul#while/lstm_cell_4/dropout_6/Mul:z:0$while/lstm_cell_4/dropout_6/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2#
!while/lstm_cell_4/dropout_6/Mul_1
!while/lstm_cell_4/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2#
!while/lstm_cell_4/dropout_7/ConstЯ
while/lstm_cell_4/dropout_7/MulMul&while/lstm_cell_4/ones_like_1:output:0*while/lstm_cell_4/dropout_7/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2!
while/lstm_cell_4/dropout_7/Mul
!while/lstm_cell_4/dropout_7/ShapeShape&while/lstm_cell_4/ones_like_1:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/dropout_7/Shape
8while/lstm_cell_4/dropout_7/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_7/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџd*
dtype0*
seedБџх)*
seed2Ж2:
8while/lstm_cell_4/dropout_7/random_uniform/RandomUniform
*while/lstm_cell_4/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2,
*while/lstm_cell_4/dropout_7/GreaterEqual/y
(while/lstm_cell_4/dropout_7/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_7/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2*
(while/lstm_cell_4/dropout_7/GreaterEqualЛ
 while/lstm_cell_4/dropout_7/CastCast,while/lstm_cell_4/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd2"
 while/lstm_cell_4/dropout_7/CastЪ
!while/lstm_cell_4/dropout_7/Mul_1Mul#while/lstm_cell_4/dropout_7/Mul:z:0$while/lstm_cell_4/dropout_7/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2#
!while/lstm_cell_4/dropout_7/Mul_1П
while/lstm_cell_4/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/lstm_cell_4/dropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_4/mulХ
while/lstm_cell_4/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_4/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_4/mul_1Х
while/lstm_cell_4/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_4/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_4/mul_2Х
while/lstm_cell_4/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_4/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_4/mul_3t
while/lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_4/Const
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_4/split/split_dimФ
&while/lstm_cell_4/split/ReadVariableOpReadVariableOp1while_lstm_cell_4_split_readvariableop_resource_0* 
_output_shapes
:
Ќ*
dtype02(
&while/lstm_cell_4/split/ReadVariableOpѓ
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0.while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	Ќd:	Ќd:	Ќd:	Ќd*
	num_split2
while/lstm_cell_4/split­
while/lstm_cell_4/MatMulMatMulwhile/lstm_cell_4/mul:z:0 while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMulГ
while/lstm_cell_4/MatMul_1MatMulwhile/lstm_cell_4/mul_1:z:0 while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMul_1Г
while/lstm_cell_4/MatMul_2MatMulwhile/lstm_cell_4/mul_2:z:0 while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMul_2Г
while/lstm_cell_4/MatMul_3MatMulwhile/lstm_cell_4/mul_3:z:0 while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMul_3x
while/lstm_cell_4/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_4/Const_1
#while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_4/split_1/split_dimХ
(while/lstm_cell_4/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02*
(while/lstm_cell_4/split_1/ReadVariableOpч
while/lstm_cell_4/split_1Split,while/lstm_cell_4/split_1/split_dim:output:00while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:d:d:d:d*
	num_split2
while/lstm_cell_4/split_1Л
while/lstm_cell_4/BiasAddBiasAdd"while/lstm_cell_4/MatMul:product:0"while/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/BiasAddС
while/lstm_cell_4/BiasAdd_1BiasAdd$while/lstm_cell_4/MatMul_1:product:0"while/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/BiasAdd_1С
while/lstm_cell_4/BiasAdd_2BiasAdd$while/lstm_cell_4/MatMul_2:product:0"while/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/BiasAdd_2С
while/lstm_cell_4/BiasAdd_3BiasAdd$while/lstm_cell_4/MatMul_3:product:0"while/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/BiasAdd_3Ї
while/lstm_cell_4/mul_4Mulwhile_placeholder_2%while/lstm_cell_4/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/mul_4Ї
while/lstm_cell_4/mul_5Mulwhile_placeholder_2%while/lstm_cell_4/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/mul_5Ї
while/lstm_cell_4/mul_6Mulwhile_placeholder_2%while/lstm_cell_4/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/mul_6Ї
while/lstm_cell_4/mul_7Mulwhile_placeholder_2%while/lstm_cell_4/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/mul_7Б
 while/lstm_cell_4/ReadVariableOpReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	d*
dtype02"
 while/lstm_cell_4/ReadVariableOp
%while/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell_4/strided_slice/stackЃ
'while/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   2)
'while/lstm_cell_4/strided_slice/stack_1Ѓ
'while/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell_4/strided_slice/stack_2ш
while/lstm_cell_4/strided_sliceStridedSlice(while/lstm_cell_4/ReadVariableOp:value:0.while/lstm_cell_4/strided_slice/stack:output:00while/lstm_cell_4/strided_slice/stack_1:output:00while/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2!
while/lstm_cell_4/strided_sliceЛ
while/lstm_cell_4/MatMul_4MatMulwhile/lstm_cell_4/mul_4:z:0(while/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMul_4Г
while/lstm_cell_4/addAddV2"while/lstm_cell_4/BiasAdd:output:0$while/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/add
while/lstm_cell_4/SigmoidSigmoidwhile/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/SigmoidЕ
"while/lstm_cell_4/ReadVariableOp_1ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	d*
dtype02$
"while/lstm_cell_4/ReadVariableOp_1Ѓ
'while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   2)
'while/lstm_cell_4/strided_slice_1/stackЇ
)while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Ш   2+
)while/lstm_cell_4/strided_slice_1/stack_1Ї
)while/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_4/strided_slice_1/stack_2є
!while/lstm_cell_4/strided_slice_1StridedSlice*while/lstm_cell_4/ReadVariableOp_1:value:00while/lstm_cell_4/strided_slice_1/stack:output:02while/lstm_cell_4/strided_slice_1/stack_1:output:02while/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_1Н
while/lstm_cell_4/MatMul_5MatMulwhile/lstm_cell_4/mul_5:z:0*while/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMul_5Й
while/lstm_cell_4/add_1AddV2$while/lstm_cell_4/BiasAdd_1:output:0$while/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/add_1
while/lstm_cell_4/Sigmoid_1Sigmoidwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/Sigmoid_1Ё
while/lstm_cell_4/mul_8Mulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/mul_8Е
"while/lstm_cell_4/ReadVariableOp_2ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	d*
dtype02$
"while/lstm_cell_4/ReadVariableOp_2Ѓ
'while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    Ш   2)
'while/lstm_cell_4/strided_slice_2/stackЇ
)while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  2+
)while/lstm_cell_4/strided_slice_2/stack_1Ї
)while/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_4/strided_slice_2/stack_2є
!while/lstm_cell_4/strided_slice_2StridedSlice*while/lstm_cell_4/ReadVariableOp_2:value:00while/lstm_cell_4/strided_slice_2/stack:output:02while/lstm_cell_4/strided_slice_2/stack_1:output:02while/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_2Н
while/lstm_cell_4/MatMul_6MatMulwhile/lstm_cell_4/mul_6:z:0*while/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMul_6Й
while/lstm_cell_4/add_2AddV2$while/lstm_cell_4/BiasAdd_2:output:0$while/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/add_2
while/lstm_cell_4/TanhTanhwhile/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/TanhІ
while/lstm_cell_4/mul_9Mulwhile/lstm_cell_4/Sigmoid:y:0while/lstm_cell_4/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/mul_9Ї
while/lstm_cell_4/add_3AddV2while/lstm_cell_4/mul_8:z:0while/lstm_cell_4/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/add_3Е
"while/lstm_cell_4/ReadVariableOp_3ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	d*
dtype02$
"while/lstm_cell_4/ReadVariableOp_3Ѓ
'while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  2)
'while/lstm_cell_4/strided_slice_3/stackЇ
)while/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_4/strided_slice_3/stack_1Ї
)while/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_4/strided_slice_3/stack_2є
!while/lstm_cell_4/strided_slice_3StridedSlice*while/lstm_cell_4/ReadVariableOp_3:value:00while/lstm_cell_4/strided_slice_3/stack:output:02while/lstm_cell_4/strided_slice_3/stack_1:output:02while/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_3Н
while/lstm_cell_4/MatMul_7MatMulwhile/lstm_cell_4/mul_7:z:0*while/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMul_7Й
while/lstm_cell_4/add_4AddV2$while/lstm_cell_4/BiasAdd_3:output:0$while/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/add_4
while/lstm_cell_4/Sigmoid_2Sigmoidwhile/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/Sigmoid_2
while/lstm_cell_4/Tanh_1Tanhwhile/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/Tanh_1Ќ
while/lstm_cell_4/mul_10Mulwhile/lstm_cell_4/Sigmoid_2:y:0while/lstm_cell_4/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/mul_10р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_4/mul_10:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_4/mul_10:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_4_readvariableop_resource+while_lstm_cell_4_readvariableop_resource_0"h
1while_lstm_cell_4_split_1_readvariableop_resource3while_lstm_cell_4_split_1_readvariableop_resource_0"d
/while_lstm_cell_4_split_readvariableop_resource1while_lstm_cell_4_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџd:џџџџџџџџџd: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
: 
Н
№
lstm_while_body_33389&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3%
!lstm_while_lstm_strided_slice_1_0a
]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0:
6lstm_while_lstm_cell_4_split_readvariableop_resource_0<
8lstm_while_lstm_cell_4_split_1_readvariableop_resource_04
0lstm_while_lstm_cell_4_readvariableop_resource_0
lstm_while_identity
lstm_while_identity_1
lstm_while_identity_2
lstm_while_identity_3
lstm_while_identity_4
lstm_while_identity_5#
lstm_while_lstm_strided_slice_1_
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor8
4lstm_while_lstm_cell_4_split_readvariableop_resource:
6lstm_while_lstm_cell_4_split_1_readvariableop_resource2
.lstm_while_lstm_cell_4_readvariableop_resourceЭ
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  2>
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeђ
.lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0lstm_while_placeholderElstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџЌ*
element_dtype020
.lstm/while/TensorArrayV2Read/TensorListGetItemЕ
&lstm/while/lstm_cell_4/ones_like/ShapeShape5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2(
&lstm/while/lstm_cell_4/ones_like/Shape
&lstm/while/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2(
&lstm/while/lstm_cell_4/ones_like/Constс
 lstm/while/lstm_cell_4/ones_likeFill/lstm/while/lstm_cell_4/ones_like/Shape:output:0/lstm/while/lstm_cell_4/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2"
 lstm/while/lstm_cell_4/ones_like
(lstm/while/lstm_cell_4/ones_like_1/ShapeShapelstm_while_placeholder_2*
T0*
_output_shapes
:2*
(lstm/while/lstm_cell_4/ones_like_1/Shape
(lstm/while/lstm_cell_4/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2*
(lstm/while/lstm_cell_4/ones_like_1/Constш
"lstm/while/lstm_cell_4/ones_like_1Fill1lstm/while/lstm_cell_4/ones_like_1/Shape:output:01lstm/while/lstm_cell_4/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2$
"lstm/while/lstm_cell_4/ones_like_1д
lstm/while/lstm_cell_4/mulMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm/while/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm/while/lstm_cell_4/mulи
lstm/while/lstm_cell_4/mul_1Mul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm/while/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm/while/lstm_cell_4/mul_1и
lstm/while/lstm_cell_4/mul_2Mul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm/while/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm/while/lstm_cell_4/mul_2и
lstm/while/lstm_cell_4/mul_3Mul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm/while/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm/while/lstm_cell_4/mul_3~
lstm/while/lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/lstm_cell_4/Const
&lstm/while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2(
&lstm/while/lstm_cell_4/split/split_dimг
+lstm/while/lstm_cell_4/split/ReadVariableOpReadVariableOp6lstm_while_lstm_cell_4_split_readvariableop_resource_0* 
_output_shapes
:
Ќ*
dtype02-
+lstm/while/lstm_cell_4/split/ReadVariableOp
lstm/while/lstm_cell_4/splitSplit/lstm/while/lstm_cell_4/split/split_dim:output:03lstm/while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	Ќd:	Ќd:	Ќd:	Ќd*
	num_split2
lstm/while/lstm_cell_4/splitС
lstm/while/lstm_cell_4/MatMulMatMullstm/while/lstm_cell_4/mul:z:0%lstm/while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/while/lstm_cell_4/MatMulЧ
lstm/while/lstm_cell_4/MatMul_1MatMul lstm/while/lstm_cell_4/mul_1:z:0%lstm/while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd2!
lstm/while/lstm_cell_4/MatMul_1Ч
lstm/while/lstm_cell_4/MatMul_2MatMul lstm/while/lstm_cell_4/mul_2:z:0%lstm/while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd2!
lstm/while/lstm_cell_4/MatMul_2Ч
lstm/while/lstm_cell_4/MatMul_3MatMul lstm/while/lstm_cell_4/mul_3:z:0%lstm/while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџd2!
lstm/while/lstm_cell_4/MatMul_3
lstm/while/lstm_cell_4/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2 
lstm/while/lstm_cell_4/Const_1
(lstm/while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(lstm/while/lstm_cell_4/split_1/split_dimд
-lstm/while/lstm_cell_4/split_1/ReadVariableOpReadVariableOp8lstm_while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02/
-lstm/while/lstm_cell_4/split_1/ReadVariableOpћ
lstm/while/lstm_cell_4/split_1Split1lstm/while/lstm_cell_4/split_1/split_dim:output:05lstm/while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:d:d:d:d*
	num_split2 
lstm/while/lstm_cell_4/split_1Я
lstm/while/lstm_cell_4/BiasAddBiasAdd'lstm/while/lstm_cell_4/MatMul:product:0'lstm/while/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2 
lstm/while/lstm_cell_4/BiasAddе
 lstm/while/lstm_cell_4/BiasAdd_1BiasAdd)lstm/while/lstm_cell_4/MatMul_1:product:0'lstm/while/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџd2"
 lstm/while/lstm_cell_4/BiasAdd_1е
 lstm/while/lstm_cell_4/BiasAdd_2BiasAdd)lstm/while/lstm_cell_4/MatMul_2:product:0'lstm/while/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџd2"
 lstm/while/lstm_cell_4/BiasAdd_2е
 lstm/while/lstm_cell_4/BiasAdd_3BiasAdd)lstm/while/lstm_cell_4/MatMul_3:product:0'lstm/while/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџd2"
 lstm/while/lstm_cell_4/BiasAdd_3М
lstm/while/lstm_cell_4/mul_4Mullstm_while_placeholder_2+lstm/while/lstm_cell_4/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/while/lstm_cell_4/mul_4М
lstm/while/lstm_cell_4/mul_5Mullstm_while_placeholder_2+lstm/while/lstm_cell_4/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/while/lstm_cell_4/mul_5М
lstm/while/lstm_cell_4/mul_6Mullstm_while_placeholder_2+lstm/while/lstm_cell_4/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/while/lstm_cell_4/mul_6М
lstm/while/lstm_cell_4/mul_7Mullstm_while_placeholder_2+lstm/while/lstm_cell_4/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/while/lstm_cell_4/mul_7Р
%lstm/while/lstm_cell_4/ReadVariableOpReadVariableOp0lstm_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	d*
dtype02'
%lstm/while/lstm_cell_4/ReadVariableOpЉ
*lstm/while/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2,
*lstm/while/lstm_cell_4/strided_slice/stack­
,lstm/while/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   2.
,lstm/while/lstm_cell_4/strided_slice/stack_1­
,lstm/while/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm/while/lstm_cell_4/strided_slice/stack_2
$lstm/while/lstm_cell_4/strided_sliceStridedSlice-lstm/while/lstm_cell_4/ReadVariableOp:value:03lstm/while/lstm_cell_4/strided_slice/stack:output:05lstm/while/lstm_cell_4/strided_slice/stack_1:output:05lstm/while/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2&
$lstm/while/lstm_cell_4/strided_sliceЯ
lstm/while/lstm_cell_4/MatMul_4MatMul lstm/while/lstm_cell_4/mul_4:z:0-lstm/while/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2!
lstm/while/lstm_cell_4/MatMul_4Ч
lstm/while/lstm_cell_4/addAddV2'lstm/while/lstm_cell_4/BiasAdd:output:0)lstm/while/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/while/lstm_cell_4/add
lstm/while/lstm_cell_4/SigmoidSigmoidlstm/while/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2 
lstm/while/lstm_cell_4/SigmoidФ
'lstm/while/lstm_cell_4/ReadVariableOp_1ReadVariableOp0lstm_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	d*
dtype02)
'lstm/while/lstm_cell_4/ReadVariableOp_1­
,lstm/while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   2.
,lstm/while/lstm_cell_4/strided_slice_1/stackБ
.lstm/while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Ш   20
.lstm/while/lstm_cell_4/strided_slice_1/stack_1Б
.lstm/while/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.lstm/while/lstm_cell_4/strided_slice_1/stack_2
&lstm/while/lstm_cell_4/strided_slice_1StridedSlice/lstm/while/lstm_cell_4/ReadVariableOp_1:value:05lstm/while/lstm_cell_4/strided_slice_1/stack:output:07lstm/while/lstm_cell_4/strided_slice_1/stack_1:output:07lstm/while/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2(
&lstm/while/lstm_cell_4/strided_slice_1б
lstm/while/lstm_cell_4/MatMul_5MatMul lstm/while/lstm_cell_4/mul_5:z:0/lstm/while/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2!
lstm/while/lstm_cell_4/MatMul_5Э
lstm/while/lstm_cell_4/add_1AddV2)lstm/while/lstm_cell_4/BiasAdd_1:output:0)lstm/while/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/while/lstm_cell_4/add_1Ѓ
 lstm/while/lstm_cell_4/Sigmoid_1Sigmoid lstm/while/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2"
 lstm/while/lstm_cell_4/Sigmoid_1Е
lstm/while/lstm_cell_4/mul_8Mul$lstm/while/lstm_cell_4/Sigmoid_1:y:0lstm_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/while/lstm_cell_4/mul_8Ф
'lstm/while/lstm_cell_4/ReadVariableOp_2ReadVariableOp0lstm_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	d*
dtype02)
'lstm/while/lstm_cell_4/ReadVariableOp_2­
,lstm/while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    Ш   2.
,lstm/while/lstm_cell_4/strided_slice_2/stackБ
.lstm/while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  20
.lstm/while/lstm_cell_4/strided_slice_2/stack_1Б
.lstm/while/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.lstm/while/lstm_cell_4/strided_slice_2/stack_2
&lstm/while/lstm_cell_4/strided_slice_2StridedSlice/lstm/while/lstm_cell_4/ReadVariableOp_2:value:05lstm/while/lstm_cell_4/strided_slice_2/stack:output:07lstm/while/lstm_cell_4/strided_slice_2/stack_1:output:07lstm/while/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2(
&lstm/while/lstm_cell_4/strided_slice_2б
lstm/while/lstm_cell_4/MatMul_6MatMul lstm/while/lstm_cell_4/mul_6:z:0/lstm/while/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2!
lstm/while/lstm_cell_4/MatMul_6Э
lstm/while/lstm_cell_4/add_2AddV2)lstm/while/lstm_cell_4/BiasAdd_2:output:0)lstm/while/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/while/lstm_cell_4/add_2
lstm/while/lstm_cell_4/TanhTanh lstm/while/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/while/lstm_cell_4/TanhК
lstm/while/lstm_cell_4/mul_9Mul"lstm/while/lstm_cell_4/Sigmoid:y:0lstm/while/lstm_cell_4/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/while/lstm_cell_4/mul_9Л
lstm/while/lstm_cell_4/add_3AddV2 lstm/while/lstm_cell_4/mul_8:z:0 lstm/while/lstm_cell_4/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/while/lstm_cell_4/add_3Ф
'lstm/while/lstm_cell_4/ReadVariableOp_3ReadVariableOp0lstm_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	d*
dtype02)
'lstm/while/lstm_cell_4/ReadVariableOp_3­
,lstm/while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  2.
,lstm/while/lstm_cell_4/strided_slice_3/stackБ
.lstm/while/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        20
.lstm/while/lstm_cell_4/strided_slice_3/stack_1Б
.lstm/while/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.lstm/while/lstm_cell_4/strided_slice_3/stack_2
&lstm/while/lstm_cell_4/strided_slice_3StridedSlice/lstm/while/lstm_cell_4/ReadVariableOp_3:value:05lstm/while/lstm_cell_4/strided_slice_3/stack:output:07lstm/while/lstm_cell_4/strided_slice_3/stack_1:output:07lstm/while/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2(
&lstm/while/lstm_cell_4/strided_slice_3б
lstm/while/lstm_cell_4/MatMul_7MatMul lstm/while/lstm_cell_4/mul_7:z:0/lstm/while/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2!
lstm/while/lstm_cell_4/MatMul_7Э
lstm/while/lstm_cell_4/add_4AddV2)lstm/while/lstm_cell_4/BiasAdd_3:output:0)lstm/while/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/while/lstm_cell_4/add_4Ѓ
 lstm/while/lstm_cell_4/Sigmoid_2Sigmoid lstm/while/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2"
 lstm/while/lstm_cell_4/Sigmoid_2
lstm/while/lstm_cell_4/Tanh_1Tanh lstm/while/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/while/lstm_cell_4/Tanh_1Р
lstm/while/lstm_cell_4/mul_10Mul$lstm/while/lstm_cell_4/Sigmoid_2:y:0!lstm/while/lstm_cell_4/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/while/lstm_cell_4/mul_10љ
/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_while_placeholder_1lstm_while_placeholder!lstm/while/lstm_cell_4/mul_10:z:0*
_output_shapes
: *
element_dtype021
/lstm/while/TensorArrayV2Write/TensorListSetItemf
lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/add/y}
lstm/while/addAddV2lstm_while_placeholderlstm/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm/while/addj
lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/add_1/y
lstm/while/add_1AddV2"lstm_while_lstm_while_loop_counterlstm/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm/while/add_1m
lstm/while/IdentityIdentitylstm/while/add_1:z:0*
T0*
_output_shapes
: 2
lstm/while/Identity
lstm/while/Identity_1Identity(lstm_while_lstm_while_maximum_iterations*
T0*
_output_shapes
: 2
lstm/while/Identity_1o
lstm/while/Identity_2Identitylstm/while/add:z:0*
T0*
_output_shapes
: 2
lstm/while/Identity_2
lstm/while/Identity_3Identity?lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
lstm/while/Identity_3
lstm/while/Identity_4Identity!lstm/while/lstm_cell_4/mul_10:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/while/Identity_4
lstm/while/Identity_5Identity lstm/while/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm/while/Identity_5"3
lstm_while_identitylstm/while/Identity:output:0"7
lstm_while_identity_1lstm/while/Identity_1:output:0"7
lstm_while_identity_2lstm/while/Identity_2:output:0"7
lstm_while_identity_3lstm/while/Identity_3:output:0"7
lstm_while_identity_4lstm/while/Identity_4:output:0"7
lstm_while_identity_5lstm/while/Identity_5:output:0"b
.lstm_while_lstm_cell_4_readvariableop_resource0lstm_while_lstm_cell_4_readvariableop_resource_0"r
6lstm_while_lstm_cell_4_split_1_readvariableop_resource8lstm_while_lstm_cell_4_split_1_readvariableop_resource_0"n
4lstm_while_lstm_cell_4_split_readvariableop_resource6lstm_while_lstm_cell_4_split_readvariableop_resource_0"D
lstm_while_lstm_strided_slice_1!lstm_while_lstm_strided_slice_1_0"М
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџd:џџџџџџџџџd: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
: 
І
О
while_cond_33765
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_33765___redundant_placeholder03
/while_while_cond_33765___redundant_placeholder13
/while_while_cond_33765___redundant_placeholder23
/while_while_cond_33765___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџd:џџџџџџџџџd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
:
В

D__inference_embedding_layer_call_and_return_conditional_losses_33576

inputs
embedding_lookup_33570
identity]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ(2
CastЮ
embedding_lookupResourceGatherembedding_lookup_33570Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/33570*,
_output_shapes
:џџџџџџџџџ(Ќ*
dtype02
embedding_lookupП
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/33570*,
_output_shapes
:џџџџџџџџџ(Ќ2
embedding_lookup/IdentityЁ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:џџџџџџџџџ(Ќ2
embedding_lookup/Identity_1}
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ(Ќ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ(::O K
'
_output_shapes
:џџџџџџџџџ(
 
_user_specified_nameinputs
ий
С
 __inference__wrapped_model_31228
embedding_input/
+sequential_embedding_embedding_lookup_30964=
9sequential_lstm_lstm_cell_4_split_readvariableop_resource?
;sequential_lstm_lstm_cell_4_split_1_readvariableop_resource7
3sequential_lstm_lstm_cell_4_readvariableop_resource9
5sequential_dense_final_matmul_readvariableop_resource:
6sequential_dense_final_biasadd_readvariableop_resource
identityЂsequential/lstm/while
sequential/embedding/CastCastembedding_input*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ(2
sequential/embedding/CastЗ
%sequential/embedding/embedding_lookupResourceGather+sequential_embedding_embedding_lookup_30964sequential/embedding/Cast:y:0*
Tindices0*>
_class4
20loc:@sequential/embedding/embedding_lookup/30964*,
_output_shapes
:џџџџџџџџџ(Ќ*
dtype02'
%sequential/embedding/embedding_lookup
.sequential/embedding/embedding_lookup/IdentityIdentity.sequential/embedding/embedding_lookup:output:0*
T0*>
_class4
20loc:@sequential/embedding/embedding_lookup/30964*,
_output_shapes
:џџџџџџџџџ(Ќ20
.sequential/embedding/embedding_lookup/Identityр
0sequential/embedding/embedding_lookup/Identity_1Identity7sequential/embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:џџџџџџџџџ(Ќ22
0sequential/embedding/embedding_lookup/Identity_1
sequential/lstm/ShapeShape9sequential/embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
sequential/lstm/Shape
#sequential/lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#sequential/lstm/strided_slice/stack
%sequential/lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%sequential/lstm/strided_slice/stack_1
%sequential/lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%sequential/lstm/strided_slice/stack_2Т
sequential/lstm/strided_sliceStridedSlicesequential/lstm/Shape:output:0,sequential/lstm/strided_slice/stack:output:0.sequential/lstm/strided_slice/stack_1:output:0.sequential/lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sequential/lstm/strided_slice|
sequential/lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
sequential/lstm/zeros/mul/yЌ
sequential/lstm/zeros/mulMul&sequential/lstm/strided_slice:output:0$sequential/lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential/lstm/zeros/mul
sequential/lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
sequential/lstm/zeros/Less/yЇ
sequential/lstm/zeros/LessLesssequential/lstm/zeros/mul:z:0%sequential/lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
sequential/lstm/zeros/Less
sequential/lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2 
sequential/lstm/zeros/packed/1У
sequential/lstm/zeros/packedPack&sequential/lstm/strided_slice:output:0'sequential/lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
sequential/lstm/zeros/packed
sequential/lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/lstm/zeros/ConstЕ
sequential/lstm/zerosFill%sequential/lstm/zeros/packed:output:0$sequential/lstm/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
sequential/lstm/zeros
sequential/lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
sequential/lstm/zeros_1/mul/yВ
sequential/lstm/zeros_1/mulMul&sequential/lstm/strided_slice:output:0&sequential/lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
sequential/lstm/zeros_1/mul
sequential/lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2 
sequential/lstm/zeros_1/Less/yЏ
sequential/lstm/zeros_1/LessLesssequential/lstm/zeros_1/mul:z:0'sequential/lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
sequential/lstm/zeros_1/Less
 sequential/lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2"
 sequential/lstm/zeros_1/packed/1Щ
sequential/lstm/zeros_1/packedPack&sequential/lstm/strided_slice:output:0)sequential/lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
sequential/lstm/zeros_1/packed
sequential/lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/lstm/zeros_1/ConstН
sequential/lstm/zeros_1Fill'sequential/lstm/zeros_1/packed:output:0&sequential/lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
sequential/lstm/zeros_1
sequential/lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
sequential/lstm/transpose/permо
sequential/lstm/transpose	Transpose9sequential/embedding/embedding_lookup/Identity_1:output:0'sequential/lstm/transpose/perm:output:0*
T0*,
_output_shapes
:(џџџџџџџџџЌ2
sequential/lstm/transpose
sequential/lstm/Shape_1Shapesequential/lstm/transpose:y:0*
T0*
_output_shapes
:2
sequential/lstm/Shape_1
%sequential/lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential/lstm/strided_slice_1/stack
'sequential/lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'sequential/lstm/strided_slice_1/stack_1
'sequential/lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'sequential/lstm/strided_slice_1/stack_2Ю
sequential/lstm/strided_slice_1StridedSlice sequential/lstm/Shape_1:output:0.sequential/lstm/strided_slice_1/stack:output:00sequential/lstm/strided_slice_1/stack_1:output:00sequential/lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
sequential/lstm/strided_slice_1Ѕ
+sequential/lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2-
+sequential/lstm/TensorArrayV2/element_shapeђ
sequential/lstm/TensorArrayV2TensorListReserve4sequential/lstm/TensorArrayV2/element_shape:output:0(sequential/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
sequential/lstm/TensorArrayV2п
Esequential/lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  2G
Esequential/lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeИ
7sequential/lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsequential/lstm/transpose:y:0Nsequential/lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7sequential/lstm/TensorArrayUnstack/TensorListFromTensor
%sequential/lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential/lstm/strided_slice_2/stack
'sequential/lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'sequential/lstm/strided_slice_2/stack_1
'sequential/lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'sequential/lstm/strided_slice_2/stack_2н
sequential/lstm/strided_slice_2StridedSlicesequential/lstm/transpose:y:0.sequential/lstm/strided_slice_2/stack:output:00sequential/lstm/strided_slice_2/stack_1:output:00sequential/lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
shrink_axis_mask2!
sequential/lstm/strided_slice_2В
+sequential/lstm/lstm_cell_4/ones_like/ShapeShape(sequential/lstm/strided_slice_2:output:0*
T0*
_output_shapes
:2-
+sequential/lstm/lstm_cell_4/ones_like/Shape
+sequential/lstm/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+sequential/lstm/lstm_cell_4/ones_like/Constѕ
%sequential/lstm/lstm_cell_4/ones_likeFill4sequential/lstm/lstm_cell_4/ones_like/Shape:output:04sequential/lstm/lstm_cell_4/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2'
%sequential/lstm/lstm_cell_4/ones_likeЌ
-sequential/lstm/lstm_cell_4/ones_like_1/ShapeShapesequential/lstm/zeros:output:0*
T0*
_output_shapes
:2/
-sequential/lstm/lstm_cell_4/ones_like_1/ShapeЃ
-sequential/lstm/lstm_cell_4/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2/
-sequential/lstm/lstm_cell_4/ones_like_1/Constќ
'sequential/lstm/lstm_cell_4/ones_like_1Fill6sequential/lstm/lstm_cell_4/ones_like_1/Shape:output:06sequential/lstm/lstm_cell_4/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2)
'sequential/lstm/lstm_cell_4/ones_like_1ж
sequential/lstm/lstm_cell_4/mulMul(sequential/lstm/strided_slice_2:output:0.sequential/lstm/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2!
sequential/lstm/lstm_cell_4/mulк
!sequential/lstm/lstm_cell_4/mul_1Mul(sequential/lstm/strided_slice_2:output:0.sequential/lstm/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2#
!sequential/lstm/lstm_cell_4/mul_1к
!sequential/lstm/lstm_cell_4/mul_2Mul(sequential/lstm/strided_slice_2:output:0.sequential/lstm/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2#
!sequential/lstm/lstm_cell_4/mul_2к
!sequential/lstm/lstm_cell_4/mul_3Mul(sequential/lstm/strided_slice_2:output:0.sequential/lstm/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2#
!sequential/lstm/lstm_cell_4/mul_3
!sequential/lstm/lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential/lstm/lstm_cell_4/Const
+sequential/lstm/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+sequential/lstm/lstm_cell_4/split/split_dimр
0sequential/lstm/lstm_cell_4/split/ReadVariableOpReadVariableOp9sequential_lstm_lstm_cell_4_split_readvariableop_resource* 
_output_shapes
:
Ќ*
dtype022
0sequential/lstm/lstm_cell_4/split/ReadVariableOp
!sequential/lstm/lstm_cell_4/splitSplit4sequential/lstm/lstm_cell_4/split/split_dim:output:08sequential/lstm/lstm_cell_4/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	Ќd:	Ќd:	Ќd:	Ќd*
	num_split2#
!sequential/lstm/lstm_cell_4/splitе
"sequential/lstm/lstm_cell_4/MatMulMatMul#sequential/lstm/lstm_cell_4/mul:z:0*sequential/lstm/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2$
"sequential/lstm/lstm_cell_4/MatMulл
$sequential/lstm/lstm_cell_4/MatMul_1MatMul%sequential/lstm/lstm_cell_4/mul_1:z:0*sequential/lstm/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd2&
$sequential/lstm/lstm_cell_4/MatMul_1л
$sequential/lstm/lstm_cell_4/MatMul_2MatMul%sequential/lstm/lstm_cell_4/mul_2:z:0*sequential/lstm/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd2&
$sequential/lstm/lstm_cell_4/MatMul_2л
$sequential/lstm/lstm_cell_4/MatMul_3MatMul%sequential/lstm/lstm_cell_4/mul_3:z:0*sequential/lstm/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџd2&
$sequential/lstm/lstm_cell_4/MatMul_3
#sequential/lstm/lstm_cell_4/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2%
#sequential/lstm/lstm_cell_4/Const_1 
-sequential/lstm/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential/lstm/lstm_cell_4/split_1/split_dimс
2sequential/lstm/lstm_cell_4/split_1/ReadVariableOpReadVariableOp;sequential_lstm_lstm_cell_4_split_1_readvariableop_resource*
_output_shapes	
:*
dtype024
2sequential/lstm/lstm_cell_4/split_1/ReadVariableOp
#sequential/lstm/lstm_cell_4/split_1Split6sequential/lstm/lstm_cell_4/split_1/split_dim:output:0:sequential/lstm/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:d:d:d:d*
	num_split2%
#sequential/lstm/lstm_cell_4/split_1у
#sequential/lstm/lstm_cell_4/BiasAddBiasAdd,sequential/lstm/lstm_cell_4/MatMul:product:0,sequential/lstm/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2%
#sequential/lstm/lstm_cell_4/BiasAddщ
%sequential/lstm/lstm_cell_4/BiasAdd_1BiasAdd.sequential/lstm/lstm_cell_4/MatMul_1:product:0,sequential/lstm/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџd2'
%sequential/lstm/lstm_cell_4/BiasAdd_1щ
%sequential/lstm/lstm_cell_4/BiasAdd_2BiasAdd.sequential/lstm/lstm_cell_4/MatMul_2:product:0,sequential/lstm/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџd2'
%sequential/lstm/lstm_cell_4/BiasAdd_2щ
%sequential/lstm/lstm_cell_4/BiasAdd_3BiasAdd.sequential/lstm/lstm_cell_4/MatMul_3:product:0,sequential/lstm/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџd2'
%sequential/lstm/lstm_cell_4/BiasAdd_3б
!sequential/lstm/lstm_cell_4/mul_4Mulsequential/lstm/zeros:output:00sequential/lstm/lstm_cell_4/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2#
!sequential/lstm/lstm_cell_4/mul_4б
!sequential/lstm/lstm_cell_4/mul_5Mulsequential/lstm/zeros:output:00sequential/lstm/lstm_cell_4/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2#
!sequential/lstm/lstm_cell_4/mul_5б
!sequential/lstm/lstm_cell_4/mul_6Mulsequential/lstm/zeros:output:00sequential/lstm/lstm_cell_4/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2#
!sequential/lstm/lstm_cell_4/mul_6б
!sequential/lstm/lstm_cell_4/mul_7Mulsequential/lstm/zeros:output:00sequential/lstm/lstm_cell_4/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2#
!sequential/lstm/lstm_cell_4/mul_7Э
*sequential/lstm/lstm_cell_4/ReadVariableOpReadVariableOp3sequential_lstm_lstm_cell_4_readvariableop_resource*
_output_shapes
:	d*
dtype02,
*sequential/lstm/lstm_cell_4/ReadVariableOpГ
/sequential/lstm/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        21
/sequential/lstm/lstm_cell_4/strided_slice/stackЗ
1sequential/lstm/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   23
1sequential/lstm/lstm_cell_4/strided_slice/stack_1З
1sequential/lstm/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      23
1sequential/lstm/lstm_cell_4/strided_slice/stack_2Є
)sequential/lstm/lstm_cell_4/strided_sliceStridedSlice2sequential/lstm/lstm_cell_4/ReadVariableOp:value:08sequential/lstm/lstm_cell_4/strided_slice/stack:output:0:sequential/lstm/lstm_cell_4/strided_slice/stack_1:output:0:sequential/lstm/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2+
)sequential/lstm/lstm_cell_4/strided_sliceу
$sequential/lstm/lstm_cell_4/MatMul_4MatMul%sequential/lstm/lstm_cell_4/mul_4:z:02sequential/lstm/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2&
$sequential/lstm/lstm_cell_4/MatMul_4л
sequential/lstm/lstm_cell_4/addAddV2,sequential/lstm/lstm_cell_4/BiasAdd:output:0.sequential/lstm/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2!
sequential/lstm/lstm_cell_4/addЌ
#sequential/lstm/lstm_cell_4/SigmoidSigmoid#sequential/lstm/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2%
#sequential/lstm/lstm_cell_4/Sigmoidб
,sequential/lstm/lstm_cell_4/ReadVariableOp_1ReadVariableOp3sequential_lstm_lstm_cell_4_readvariableop_resource*
_output_shapes
:	d*
dtype02.
,sequential/lstm/lstm_cell_4/ReadVariableOp_1З
1sequential/lstm/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   23
1sequential/lstm/lstm_cell_4/strided_slice_1/stackЛ
3sequential/lstm/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Ш   25
3sequential/lstm/lstm_cell_4/strided_slice_1/stack_1Л
3sequential/lstm/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      25
3sequential/lstm/lstm_cell_4/strided_slice_1/stack_2А
+sequential/lstm/lstm_cell_4/strided_slice_1StridedSlice4sequential/lstm/lstm_cell_4/ReadVariableOp_1:value:0:sequential/lstm/lstm_cell_4/strided_slice_1/stack:output:0<sequential/lstm/lstm_cell_4/strided_slice_1/stack_1:output:0<sequential/lstm/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2-
+sequential/lstm/lstm_cell_4/strided_slice_1х
$sequential/lstm/lstm_cell_4/MatMul_5MatMul%sequential/lstm/lstm_cell_4/mul_5:z:04sequential/lstm/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2&
$sequential/lstm/lstm_cell_4/MatMul_5с
!sequential/lstm/lstm_cell_4/add_1AddV2.sequential/lstm/lstm_cell_4/BiasAdd_1:output:0.sequential/lstm/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2#
!sequential/lstm/lstm_cell_4/add_1В
%sequential/lstm/lstm_cell_4/Sigmoid_1Sigmoid%sequential/lstm/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2'
%sequential/lstm/lstm_cell_4/Sigmoid_1Ь
!sequential/lstm/lstm_cell_4/mul_8Mul)sequential/lstm/lstm_cell_4/Sigmoid_1:y:0 sequential/lstm/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2#
!sequential/lstm/lstm_cell_4/mul_8б
,sequential/lstm/lstm_cell_4/ReadVariableOp_2ReadVariableOp3sequential_lstm_lstm_cell_4_readvariableop_resource*
_output_shapes
:	d*
dtype02.
,sequential/lstm/lstm_cell_4/ReadVariableOp_2З
1sequential/lstm/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    Ш   23
1sequential/lstm/lstm_cell_4/strided_slice_2/stackЛ
3sequential/lstm/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  25
3sequential/lstm/lstm_cell_4/strided_slice_2/stack_1Л
3sequential/lstm/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      25
3sequential/lstm/lstm_cell_4/strided_slice_2/stack_2А
+sequential/lstm/lstm_cell_4/strided_slice_2StridedSlice4sequential/lstm/lstm_cell_4/ReadVariableOp_2:value:0:sequential/lstm/lstm_cell_4/strided_slice_2/stack:output:0<sequential/lstm/lstm_cell_4/strided_slice_2/stack_1:output:0<sequential/lstm/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2-
+sequential/lstm/lstm_cell_4/strided_slice_2х
$sequential/lstm/lstm_cell_4/MatMul_6MatMul%sequential/lstm/lstm_cell_4/mul_6:z:04sequential/lstm/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2&
$sequential/lstm/lstm_cell_4/MatMul_6с
!sequential/lstm/lstm_cell_4/add_2AddV2.sequential/lstm/lstm_cell_4/BiasAdd_2:output:0.sequential/lstm/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2#
!sequential/lstm/lstm_cell_4/add_2Ѕ
 sequential/lstm/lstm_cell_4/TanhTanh%sequential/lstm/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2"
 sequential/lstm/lstm_cell_4/TanhЮ
!sequential/lstm/lstm_cell_4/mul_9Mul'sequential/lstm/lstm_cell_4/Sigmoid:y:0$sequential/lstm/lstm_cell_4/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2#
!sequential/lstm/lstm_cell_4/mul_9Я
!sequential/lstm/lstm_cell_4/add_3AddV2%sequential/lstm/lstm_cell_4/mul_8:z:0%sequential/lstm/lstm_cell_4/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2#
!sequential/lstm/lstm_cell_4/add_3б
,sequential/lstm/lstm_cell_4/ReadVariableOp_3ReadVariableOp3sequential_lstm_lstm_cell_4_readvariableop_resource*
_output_shapes
:	d*
dtype02.
,sequential/lstm/lstm_cell_4/ReadVariableOp_3З
1sequential/lstm/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  23
1sequential/lstm/lstm_cell_4/strided_slice_3/stackЛ
3sequential/lstm/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        25
3sequential/lstm/lstm_cell_4/strided_slice_3/stack_1Л
3sequential/lstm/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      25
3sequential/lstm/lstm_cell_4/strided_slice_3/stack_2А
+sequential/lstm/lstm_cell_4/strided_slice_3StridedSlice4sequential/lstm/lstm_cell_4/ReadVariableOp_3:value:0:sequential/lstm/lstm_cell_4/strided_slice_3/stack:output:0<sequential/lstm/lstm_cell_4/strided_slice_3/stack_1:output:0<sequential/lstm/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2-
+sequential/lstm/lstm_cell_4/strided_slice_3х
$sequential/lstm/lstm_cell_4/MatMul_7MatMul%sequential/lstm/lstm_cell_4/mul_7:z:04sequential/lstm/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2&
$sequential/lstm/lstm_cell_4/MatMul_7с
!sequential/lstm/lstm_cell_4/add_4AddV2.sequential/lstm/lstm_cell_4/BiasAdd_3:output:0.sequential/lstm/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2#
!sequential/lstm/lstm_cell_4/add_4В
%sequential/lstm/lstm_cell_4/Sigmoid_2Sigmoid%sequential/lstm/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2'
%sequential/lstm/lstm_cell_4/Sigmoid_2Љ
"sequential/lstm/lstm_cell_4/Tanh_1Tanh%sequential/lstm/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2$
"sequential/lstm/lstm_cell_4/Tanh_1д
"sequential/lstm/lstm_cell_4/mul_10Mul)sequential/lstm/lstm_cell_4/Sigmoid_2:y:0&sequential/lstm/lstm_cell_4/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2$
"sequential/lstm/lstm_cell_4/mul_10Џ
-sequential/lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   2/
-sequential/lstm/TensorArrayV2_1/element_shapeј
sequential/lstm/TensorArrayV2_1TensorListReserve6sequential/lstm/TensorArrayV2_1/element_shape:output:0(sequential/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
sequential/lstm/TensorArrayV2_1n
sequential/lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential/lstm/time
(sequential/lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2*
(sequential/lstm/while/maximum_iterations
"sequential/lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"sequential/lstm/while/loop_counterЯ
sequential/lstm/whileWhile+sequential/lstm/while/loop_counter:output:01sequential/lstm/while/maximum_iterations:output:0sequential/lstm/time:output:0(sequential/lstm/TensorArrayV2_1:handle:0sequential/lstm/zeros:output:0 sequential/lstm/zeros_1:output:0(sequential/lstm/strided_slice_1:output:0Gsequential/lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:09sequential_lstm_lstm_cell_4_split_readvariableop_resource;sequential_lstm_lstm_cell_4_split_1_readvariableop_resource3sequential_lstm_lstm_cell_4_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *%
_read_only_resource_inputs
	
*,
body$R"
 sequential_lstm_while_body_31085*,
cond$R"
 sequential_lstm_while_cond_31084*K
output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *
parallel_iterations 2
sequential/lstm/whileе
@sequential/lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   2B
@sequential/lstm/TensorArrayV2Stack/TensorListStack/element_shapeЈ
2sequential/lstm/TensorArrayV2Stack/TensorListStackTensorListStacksequential/lstm/while:output:3Isequential/lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:(џџџџџџџџџd*
element_dtype024
2sequential/lstm/TensorArrayV2Stack/TensorListStackЁ
%sequential/lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2'
%sequential/lstm/strided_slice_3/stack
'sequential/lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential/lstm/strided_slice_3/stack_1
'sequential/lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'sequential/lstm/strided_slice_3/stack_2њ
sequential/lstm/strided_slice_3StridedSlice;sequential/lstm/TensorArrayV2Stack/TensorListStack:tensor:0.sequential/lstm/strided_slice_3/stack:output:00sequential/lstm/strided_slice_3/stack_1:output:00sequential/lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_mask2!
sequential/lstm/strided_slice_3
 sequential/lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 sequential/lstm/transpose_1/permх
sequential/lstm/transpose_1	Transpose;sequential/lstm/TensorArrayV2Stack/TensorListStack:tensor:0)sequential/lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ(d2
sequential/lstm/transpose_1
sequential/lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/lstm/runtimeв
,sequential/dense_final/MatMul/ReadVariableOpReadVariableOp5sequential_dense_final_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02.
,sequential/dense_final/MatMul/ReadVariableOpк
sequential/dense_final/MatMulMatMul(sequential/lstm/strided_slice_3:output:04sequential/dense_final/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential/dense_final/MatMulб
-sequential/dense_final/BiasAdd/ReadVariableOpReadVariableOp6sequential_dense_final_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential/dense_final/BiasAdd/ReadVariableOpн
sequential/dense_final/BiasAddBiasAdd'sequential/dense_final/MatMul:product:05sequential/dense_final/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2 
sequential/dense_final/BiasAddІ
sequential/dense_final/SoftmaxSoftmax'sequential/dense_final/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2 
sequential/dense_final/Softmax
IdentityIdentity(sequential/dense_final/Softmax:softmax:0^sequential/lstm/while*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ(::::::2.
sequential/lstm/whilesequential/lstm/while:X T
'
_output_shapes
:џџџџџџџџџ(
)
_user_specified_nameembedding_input
эH

F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_35173

inputs
states_0
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2X
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like/Const
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
	ones_like^
ones_like_1/ShapeShapestates_0*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like_1/Const
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
ones_like_1`
mulMulinputsones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
muld
mul_1Mulinputsones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
mul_1d
mul_2Mulinputsones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
mul_2d
mul_3Mulinputsones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
mul_3P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
Ќ*
dtype02
split/ReadVariableOpЋ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	Ќd:	Ќd:	Ќd:	Ќd*
	num_split2
splite
MatMulMatMulmul:z:0split:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
MatMulk
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd2

MatMul_1k
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd2

MatMul_2k
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*'
_output_shapes
:џџџџџџџџџd2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype02
split_1/ReadVariableOp
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:d:d:d:d*
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџd2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџd2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџd2
	BiasAdd_3g
mul_4Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
mul_4g
mul_5Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
mul_5g
mul_6Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
mul_6g
mul_7Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
mul_7y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	d*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2ќ
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2
strided_slices
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2	
Sigmoid}
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	d*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Ш   2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2
strided_slice_1u
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2

MatMul_5q
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
	Sigmoid_1`
mul_8MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџd2
mul_8}
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	d*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    Ш   2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2
strided_slice_2u
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2

MatMul_6q
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
Tanh^
mul_9MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
mul_9_
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
add_3}
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	d*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  2
strided_slice_3/stack
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2
strided_slice_3u
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2

MatMul_7q
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
	Sigmoid_2U
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
Tanh_1d
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
mul_10^
IdentityIdentity
mul_10:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2

Identityb

Identity_1Identity
mul_10:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity_1a

Identity_2Identity	add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:џџџџџџџџџЌ:џџџџџџџџџd:џџџџџџџџџd::::P L
(
_output_shapes
:џџџџџџџџџЌ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџd
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџd
"
_user_specified_name
states/1
э$

while_body_31794
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_4_31818_0
while_lstm_cell_4_31820_0
while_lstm_cell_4_31822_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_4_31818
while_lstm_cell_4_31820
while_lstm_cell_4_31822Ђ)while/lstm_cell_4/StatefulPartitionedCallУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeд
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџЌ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemз
)while/lstm_cell_4/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_4_31818_0while_lstm_cell_4_31820_0while_lstm_cell_4_31822_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_314162+
)while/lstm_cell_4/StatefulPartitionedCallі
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_4/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2Й
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Т
while/Identity_4Identity2while/lstm_cell_4/StatefulPartitionedCall:output:1*^while/lstm_cell_4/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџd2
while/Identity_4Т
while/Identity_5Identity2while/lstm_cell_4/StatefulPartitionedCall:output:2*^while/lstm_cell_4/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџd2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_4_31818while_lstm_cell_4_31818_0"4
while_lstm_cell_4_31820while_lstm_cell_4_31820_0"4
while_lstm_cell_4_31822while_lstm_cell_4_31822_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџd:џџџџџџџџџd: : :::2V
)while/lstm_cell_4/StatefulPartitionedCall)while/lstm_cell_4/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
: 
х
п
while_body_34426
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_05
1while_lstm_cell_4_split_readvariableop_resource_07
3while_lstm_cell_4_split_1_readvariableop_resource_0/
+while_lstm_cell_4_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor3
/while_lstm_cell_4_split_readvariableop_resource5
1while_lstm_cell_4_split_1_readvariableop_resource-
)while_lstm_cell_4_readvariableop_resourceУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeд
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџЌ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemІ
!while/lstm_cell_4/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/ones_like/Shape
!while/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!while/lstm_cell_4/ones_like/ConstЭ
while/lstm_cell_4/ones_likeFill*while/lstm_cell_4/ones_like/Shape:output:0*while/lstm_cell_4/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_4/ones_like
while/lstm_cell_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2!
while/lstm_cell_4/dropout/ConstШ
while/lstm_cell_4/dropout/MulMul$while/lstm_cell_4/ones_like:output:0(while/lstm_cell_4/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_4/dropout/Mul
while/lstm_cell_4/dropout/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell_4/dropout/Shape
6while/lstm_cell_4/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_4/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*
seedБџх)*
seed2љ28
6while/lstm_cell_4/dropout/random_uniform/RandomUniform
(while/lstm_cell_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2*
(while/lstm_cell_4/dropout/GreaterEqual/y
&while/lstm_cell_4/dropout/GreaterEqualGreaterEqual?while/lstm_cell_4/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_4/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2(
&while/lstm_cell_4/dropout/GreaterEqualЖ
while/lstm_cell_4/dropout/CastCast*while/lstm_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2 
while/lstm_cell_4/dropout/CastУ
while/lstm_cell_4/dropout/Mul_1Mul!while/lstm_cell_4/dropout/Mul:z:0"while/lstm_cell_4/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2!
while/lstm_cell_4/dropout/Mul_1
!while/lstm_cell_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2#
!while/lstm_cell_4/dropout_1/ConstЮ
while/lstm_cell_4/dropout_1/MulMul$while/lstm_cell_4/ones_like:output:0*while/lstm_cell_4/dropout_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2!
while/lstm_cell_4/dropout_1/Mul
!while/lstm_cell_4/dropout_1/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/dropout_1/Shape
8while/lstm_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_1/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*
seedБџх)*
seed2ЗГ*2:
8while/lstm_cell_4/dropout_1/random_uniform/RandomUniform
*while/lstm_cell_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2,
*while/lstm_cell_4/dropout_1/GreaterEqual/y
(while/lstm_cell_4/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2*
(while/lstm_cell_4/dropout_1/GreaterEqualМ
 while/lstm_cell_4/dropout_1/CastCast,while/lstm_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2"
 while/lstm_cell_4/dropout_1/CastЫ
!while/lstm_cell_4/dropout_1/Mul_1Mul#while/lstm_cell_4/dropout_1/Mul:z:0$while/lstm_cell_4/dropout_1/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2#
!while/lstm_cell_4/dropout_1/Mul_1
!while/lstm_cell_4/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2#
!while/lstm_cell_4/dropout_2/ConstЮ
while/lstm_cell_4/dropout_2/MulMul$while/lstm_cell_4/ones_like:output:0*while/lstm_cell_4/dropout_2/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2!
while/lstm_cell_4/dropout_2/Mul
!while/lstm_cell_4/dropout_2/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/dropout_2/Shape
8while/lstm_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_2/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*
seedБџх)*
seed2ђ§ё2:
8while/lstm_cell_4/dropout_2/random_uniform/RandomUniform
*while/lstm_cell_4/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2,
*while/lstm_cell_4/dropout_2/GreaterEqual/y
(while/lstm_cell_4/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2*
(while/lstm_cell_4/dropout_2/GreaterEqualМ
 while/lstm_cell_4/dropout_2/CastCast,while/lstm_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2"
 while/lstm_cell_4/dropout_2/CastЫ
!while/lstm_cell_4/dropout_2/Mul_1Mul#while/lstm_cell_4/dropout_2/Mul:z:0$while/lstm_cell_4/dropout_2/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2#
!while/lstm_cell_4/dropout_2/Mul_1
!while/lstm_cell_4/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2#
!while/lstm_cell_4/dropout_3/ConstЮ
while/lstm_cell_4/dropout_3/MulMul$while/lstm_cell_4/ones_like:output:0*while/lstm_cell_4/dropout_3/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2!
while/lstm_cell_4/dropout_3/Mul
!while/lstm_cell_4/dropout_3/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/dropout_3/Shape
8while/lstm_cell_4/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_3/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*
seedБџх)*
seed2Л2:
8while/lstm_cell_4/dropout_3/random_uniform/RandomUniform
*while/lstm_cell_4/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2,
*while/lstm_cell_4/dropout_3/GreaterEqual/y
(while/lstm_cell_4/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2*
(while/lstm_cell_4/dropout_3/GreaterEqualМ
 while/lstm_cell_4/dropout_3/CastCast,while/lstm_cell_4/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2"
 while/lstm_cell_4/dropout_3/CastЫ
!while/lstm_cell_4/dropout_3/Mul_1Mul#while/lstm_cell_4/dropout_3/Mul:z:0$while/lstm_cell_4/dropout_3/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2#
!while/lstm_cell_4/dropout_3/Mul_1
#while/lstm_cell_4/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2%
#while/lstm_cell_4/ones_like_1/Shape
#while/lstm_cell_4/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2%
#while/lstm_cell_4/ones_like_1/Constд
while/lstm_cell_4/ones_like_1Fill,while/lstm_cell_4/ones_like_1/Shape:output:0,while/lstm_cell_4/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/ones_like_1
!while/lstm_cell_4/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2#
!while/lstm_cell_4/dropout_4/ConstЯ
while/lstm_cell_4/dropout_4/MulMul&while/lstm_cell_4/ones_like_1:output:0*while/lstm_cell_4/dropout_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2!
while/lstm_cell_4/dropout_4/Mul
!while/lstm_cell_4/dropout_4/ShapeShape&while/lstm_cell_4/ones_like_1:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/dropout_4/Shape
8while/lstm_cell_4/dropout_4/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџd*
dtype0*
seedБџх)*
seed2ЭЅ2:
8while/lstm_cell_4/dropout_4/random_uniform/RandomUniform
*while/lstm_cell_4/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2,
*while/lstm_cell_4/dropout_4/GreaterEqual/y
(while/lstm_cell_4/dropout_4/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_4/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2*
(while/lstm_cell_4/dropout_4/GreaterEqualЛ
 while/lstm_cell_4/dropout_4/CastCast,while/lstm_cell_4/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd2"
 while/lstm_cell_4/dropout_4/CastЪ
!while/lstm_cell_4/dropout_4/Mul_1Mul#while/lstm_cell_4/dropout_4/Mul:z:0$while/lstm_cell_4/dropout_4/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2#
!while/lstm_cell_4/dropout_4/Mul_1
!while/lstm_cell_4/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2#
!while/lstm_cell_4/dropout_5/ConstЯ
while/lstm_cell_4/dropout_5/MulMul&while/lstm_cell_4/ones_like_1:output:0*while/lstm_cell_4/dropout_5/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2!
while/lstm_cell_4/dropout_5/Mul
!while/lstm_cell_4/dropout_5/ShapeShape&while/lstm_cell_4/ones_like_1:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/dropout_5/Shape
8while/lstm_cell_4/dropout_5/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџd*
dtype0*
seedБџх)*
seed2ізЦ2:
8while/lstm_cell_4/dropout_5/random_uniform/RandomUniform
*while/lstm_cell_4/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2,
*while/lstm_cell_4/dropout_5/GreaterEqual/y
(while/lstm_cell_4/dropout_5/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_5/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2*
(while/lstm_cell_4/dropout_5/GreaterEqualЛ
 while/lstm_cell_4/dropout_5/CastCast,while/lstm_cell_4/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd2"
 while/lstm_cell_4/dropout_5/CastЪ
!while/lstm_cell_4/dropout_5/Mul_1Mul#while/lstm_cell_4/dropout_5/Mul:z:0$while/lstm_cell_4/dropout_5/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2#
!while/lstm_cell_4/dropout_5/Mul_1
!while/lstm_cell_4/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2#
!while/lstm_cell_4/dropout_6/ConstЯ
while/lstm_cell_4/dropout_6/MulMul&while/lstm_cell_4/ones_like_1:output:0*while/lstm_cell_4/dropout_6/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2!
while/lstm_cell_4/dropout_6/Mul
!while/lstm_cell_4/dropout_6/ShapeShape&while/lstm_cell_4/ones_like_1:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/dropout_6/Shape
8while/lstm_cell_4/dropout_6/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_6/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџd*
dtype0*
seedБџх)*
seed2Ц\2:
8while/lstm_cell_4/dropout_6/random_uniform/RandomUniform
*while/lstm_cell_4/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2,
*while/lstm_cell_4/dropout_6/GreaterEqual/y
(while/lstm_cell_4/dropout_6/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_6/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2*
(while/lstm_cell_4/dropout_6/GreaterEqualЛ
 while/lstm_cell_4/dropout_6/CastCast,while/lstm_cell_4/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd2"
 while/lstm_cell_4/dropout_6/CastЪ
!while/lstm_cell_4/dropout_6/Mul_1Mul#while/lstm_cell_4/dropout_6/Mul:z:0$while/lstm_cell_4/dropout_6/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2#
!while/lstm_cell_4/dropout_6/Mul_1
!while/lstm_cell_4/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2#
!while/lstm_cell_4/dropout_7/ConstЯ
while/lstm_cell_4/dropout_7/MulMul&while/lstm_cell_4/ones_like_1:output:0*while/lstm_cell_4/dropout_7/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2!
while/lstm_cell_4/dropout_7/Mul
!while/lstm_cell_4/dropout_7/ShapeShape&while/lstm_cell_4/ones_like_1:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/dropout_7/Shape
8while/lstm_cell_4/dropout_7/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_7/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџd*
dtype0*
seedБџх)*
seed2сў2:
8while/lstm_cell_4/dropout_7/random_uniform/RandomUniform
*while/lstm_cell_4/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2,
*while/lstm_cell_4/dropout_7/GreaterEqual/y
(while/lstm_cell_4/dropout_7/GreaterEqualGreaterEqualAwhile/lstm_cell_4/dropout_7/random_uniform/RandomUniform:output:03while/lstm_cell_4/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2*
(while/lstm_cell_4/dropout_7/GreaterEqualЛ
 while/lstm_cell_4/dropout_7/CastCast,while/lstm_cell_4/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd2"
 while/lstm_cell_4/dropout_7/CastЪ
!while/lstm_cell_4/dropout_7/Mul_1Mul#while/lstm_cell_4/dropout_7/Mul:z:0$while/lstm_cell_4/dropout_7/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2#
!while/lstm_cell_4/dropout_7/Mul_1П
while/lstm_cell_4/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/lstm_cell_4/dropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_4/mulХ
while/lstm_cell_4/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_4/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_4/mul_1Х
while/lstm_cell_4/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_4/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_4/mul_2Х
while/lstm_cell_4/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_4/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_4/mul_3t
while/lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_4/Const
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_4/split/split_dimФ
&while/lstm_cell_4/split/ReadVariableOpReadVariableOp1while_lstm_cell_4_split_readvariableop_resource_0* 
_output_shapes
:
Ќ*
dtype02(
&while/lstm_cell_4/split/ReadVariableOpѓ
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0.while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	Ќd:	Ќd:	Ќd:	Ќd*
	num_split2
while/lstm_cell_4/split­
while/lstm_cell_4/MatMulMatMulwhile/lstm_cell_4/mul:z:0 while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMulГ
while/lstm_cell_4/MatMul_1MatMulwhile/lstm_cell_4/mul_1:z:0 while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMul_1Г
while/lstm_cell_4/MatMul_2MatMulwhile/lstm_cell_4/mul_2:z:0 while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMul_2Г
while/lstm_cell_4/MatMul_3MatMulwhile/lstm_cell_4/mul_3:z:0 while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMul_3x
while/lstm_cell_4/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_4/Const_1
#while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_4/split_1/split_dimХ
(while/lstm_cell_4/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02*
(while/lstm_cell_4/split_1/ReadVariableOpч
while/lstm_cell_4/split_1Split,while/lstm_cell_4/split_1/split_dim:output:00while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:d:d:d:d*
	num_split2
while/lstm_cell_4/split_1Л
while/lstm_cell_4/BiasAddBiasAdd"while/lstm_cell_4/MatMul:product:0"while/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/BiasAddС
while/lstm_cell_4/BiasAdd_1BiasAdd$while/lstm_cell_4/MatMul_1:product:0"while/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/BiasAdd_1С
while/lstm_cell_4/BiasAdd_2BiasAdd$while/lstm_cell_4/MatMul_2:product:0"while/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/BiasAdd_2С
while/lstm_cell_4/BiasAdd_3BiasAdd$while/lstm_cell_4/MatMul_3:product:0"while/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/BiasAdd_3Ї
while/lstm_cell_4/mul_4Mulwhile_placeholder_2%while/lstm_cell_4/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/mul_4Ї
while/lstm_cell_4/mul_5Mulwhile_placeholder_2%while/lstm_cell_4/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/mul_5Ї
while/lstm_cell_4/mul_6Mulwhile_placeholder_2%while/lstm_cell_4/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/mul_6Ї
while/lstm_cell_4/mul_7Mulwhile_placeholder_2%while/lstm_cell_4/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/mul_7Б
 while/lstm_cell_4/ReadVariableOpReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	d*
dtype02"
 while/lstm_cell_4/ReadVariableOp
%while/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell_4/strided_slice/stackЃ
'while/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   2)
'while/lstm_cell_4/strided_slice/stack_1Ѓ
'while/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell_4/strided_slice/stack_2ш
while/lstm_cell_4/strided_sliceStridedSlice(while/lstm_cell_4/ReadVariableOp:value:0.while/lstm_cell_4/strided_slice/stack:output:00while/lstm_cell_4/strided_slice/stack_1:output:00while/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2!
while/lstm_cell_4/strided_sliceЛ
while/lstm_cell_4/MatMul_4MatMulwhile/lstm_cell_4/mul_4:z:0(while/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMul_4Г
while/lstm_cell_4/addAddV2"while/lstm_cell_4/BiasAdd:output:0$while/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/add
while/lstm_cell_4/SigmoidSigmoidwhile/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/SigmoidЕ
"while/lstm_cell_4/ReadVariableOp_1ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	d*
dtype02$
"while/lstm_cell_4/ReadVariableOp_1Ѓ
'while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   2)
'while/lstm_cell_4/strided_slice_1/stackЇ
)while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Ш   2+
)while/lstm_cell_4/strided_slice_1/stack_1Ї
)while/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_4/strided_slice_1/stack_2є
!while/lstm_cell_4/strided_slice_1StridedSlice*while/lstm_cell_4/ReadVariableOp_1:value:00while/lstm_cell_4/strided_slice_1/stack:output:02while/lstm_cell_4/strided_slice_1/stack_1:output:02while/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_1Н
while/lstm_cell_4/MatMul_5MatMulwhile/lstm_cell_4/mul_5:z:0*while/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMul_5Й
while/lstm_cell_4/add_1AddV2$while/lstm_cell_4/BiasAdd_1:output:0$while/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/add_1
while/lstm_cell_4/Sigmoid_1Sigmoidwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/Sigmoid_1Ё
while/lstm_cell_4/mul_8Mulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/mul_8Е
"while/lstm_cell_4/ReadVariableOp_2ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	d*
dtype02$
"while/lstm_cell_4/ReadVariableOp_2Ѓ
'while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    Ш   2)
'while/lstm_cell_4/strided_slice_2/stackЇ
)while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  2+
)while/lstm_cell_4/strided_slice_2/stack_1Ї
)while/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_4/strided_slice_2/stack_2є
!while/lstm_cell_4/strided_slice_2StridedSlice*while/lstm_cell_4/ReadVariableOp_2:value:00while/lstm_cell_4/strided_slice_2/stack:output:02while/lstm_cell_4/strided_slice_2/stack_1:output:02while/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_2Н
while/lstm_cell_4/MatMul_6MatMulwhile/lstm_cell_4/mul_6:z:0*while/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMul_6Й
while/lstm_cell_4/add_2AddV2$while/lstm_cell_4/BiasAdd_2:output:0$while/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/add_2
while/lstm_cell_4/TanhTanhwhile/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/TanhІ
while/lstm_cell_4/mul_9Mulwhile/lstm_cell_4/Sigmoid:y:0while/lstm_cell_4/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/mul_9Ї
while/lstm_cell_4/add_3AddV2while/lstm_cell_4/mul_8:z:0while/lstm_cell_4/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/add_3Е
"while/lstm_cell_4/ReadVariableOp_3ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	d*
dtype02$
"while/lstm_cell_4/ReadVariableOp_3Ѓ
'while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  2)
'while/lstm_cell_4/strided_slice_3/stackЇ
)while/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_4/strided_slice_3/stack_1Ї
)while/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_4/strided_slice_3/stack_2є
!while/lstm_cell_4/strided_slice_3StridedSlice*while/lstm_cell_4/ReadVariableOp_3:value:00while/lstm_cell_4/strided_slice_3/stack:output:02while/lstm_cell_4/strided_slice_3/stack_1:output:02while/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_3Н
while/lstm_cell_4/MatMul_7MatMulwhile/lstm_cell_4/mul_7:z:0*while/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMul_7Й
while/lstm_cell_4/add_4AddV2$while/lstm_cell_4/BiasAdd_3:output:0$while/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/add_4
while/lstm_cell_4/Sigmoid_2Sigmoidwhile/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/Sigmoid_2
while/lstm_cell_4/Tanh_1Tanhwhile/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/Tanh_1Ќ
while/lstm_cell_4/mul_10Mulwhile/lstm_cell_4/Sigmoid_2:y:0while/lstm_cell_4/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/mul_10р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_4/mul_10:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_4/mul_10:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_4_readvariableop_resource+while_lstm_cell_4_readvariableop_resource_0"h
1while_lstm_cell_4_split_1_readvariableop_resource3while_lstm_cell_4_split_1_readvariableop_resource_0"d
/while_lstm_cell_4_split_readvariableop_resource1while_lstm_cell_4_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџd:џџџџџџџџџd: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
: 
џ

F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_35089

inputs
states_0
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2X
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like/Const
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2
dropout/Const
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/Shapeд
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*
seedБџх)*
seed2ѓќ2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2
dropout/GreaterEqual/yП
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2
dropout_1/Const
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/Shapeй
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*
seedБџх)*
seed2f2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2
dropout_1/GreaterEqual/yЧ
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
dropout_1/GreaterEqual
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
dropout_1/Cast
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2
dropout_2/Const
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/Shapeк
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*
seedБџх)*
seed2џЫ2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2
dropout_2/GreaterEqual/yЧ
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
dropout_2/GreaterEqual
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
dropout_2/Cast
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
dropout_2/Mul_1g
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2
dropout_3/Const
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
dropout_3/Muld
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_3/Shapeк
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*
seedБџх)*
seed2ѕл2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2
dropout_3/GreaterEqual/yЧ
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
dropout_3/GreaterEqual
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
dropout_3/Cast
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
dropout_3/Mul_1^
ones_like_1/ShapeShapestates_0*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like_1/Const
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
ones_like_1g
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2
dropout_4/Const
dropout_4/MulMulones_like_1:output:0dropout_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
dropout_4/Mulf
dropout_4/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_4/Shapeй
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџd*
dtype0*
seedБџх)*
seed2ы2(
&dropout_4/random_uniform/RandomUniformy
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2
dropout_4/GreaterEqual/yЦ
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
dropout_4/GreaterEqual
dropout_4/CastCastdropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd2
dropout_4/Cast
dropout_4/Mul_1Muldropout_4/Mul:z:0dropout_4/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
dropout_4/Mul_1g
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2
dropout_5/Const
dropout_5/MulMulones_like_1:output:0dropout_5/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
dropout_5/Mulf
dropout_5/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_5/Shapeй
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџd*
dtype0*
seedБџх)*
seed2њм2(
&dropout_5/random_uniform/RandomUniformy
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2
dropout_5/GreaterEqual/yЦ
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
dropout_5/GreaterEqual
dropout_5/CastCastdropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd2
dropout_5/Cast
dropout_5/Mul_1Muldropout_5/Mul:z:0dropout_5/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
dropout_5/Mul_1g
dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2
dropout_6/Const
dropout_6/MulMulones_like_1:output:0dropout_6/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
dropout_6/Mulf
dropout_6/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_6/Shapeй
&dropout_6/random_uniform/RandomUniformRandomUniformdropout_6/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџd*
dtype0*
seedБџх)*
seed2њљЇ2(
&dropout_6/random_uniform/RandomUniformy
dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2
dropout_6/GreaterEqual/yЦ
dropout_6/GreaterEqualGreaterEqual/dropout_6/random_uniform/RandomUniform:output:0!dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
dropout_6/GreaterEqual
dropout_6/CastCastdropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd2
dropout_6/Cast
dropout_6/Mul_1Muldropout_6/Mul:z:0dropout_6/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
dropout_6/Mul_1g
dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2
dropout_7/Const
dropout_7/MulMulones_like_1:output:0dropout_7/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
dropout_7/Mulf
dropout_7/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_7/Shapeй
&dropout_7/random_uniform/RandomUniformRandomUniformdropout_7/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџd*
dtype0*
seedБџх)*
seed2М­2(
&dropout_7/random_uniform/RandomUniformy
dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2
dropout_7/GreaterEqual/yЦ
dropout_7/GreaterEqualGreaterEqual/dropout_7/random_uniform/RandomUniform:output:0!dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
dropout_7/GreaterEqual
dropout_7/CastCastdropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd2
dropout_7/Cast
dropout_7/Mul_1Muldropout_7/Mul:z:0dropout_7/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
dropout_7/Mul_1_
mulMulinputsdropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
mule
mul_1Mulinputsdropout_1/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
mul_1e
mul_2Mulinputsdropout_2/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
mul_2e
mul_3Mulinputsdropout_3/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
mul_3P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
Ќ*
dtype02
split/ReadVariableOpЋ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	Ќd:	Ќd:	Ќd:	Ќd*
	num_split2
splite
MatMulMatMulmul:z:0split:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
MatMulk
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd2

MatMul_1k
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd2

MatMul_2k
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*'
_output_shapes
:џџџџџџџџџd2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype02
split_1/ReadVariableOp
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:d:d:d:d*
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџd2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџd2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџd2
	BiasAdd_3f
mul_4Mulstates_0dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
mul_4f
mul_5Mulstates_0dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
mul_5f
mul_6Mulstates_0dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
mul_6f
mul_7Mulstates_0dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
mul_7y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	d*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2ќ
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2
strided_slices
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2	
Sigmoid}
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	d*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Ш   2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2
strided_slice_1u
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2

MatMul_5q
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
	Sigmoid_1`
mul_8MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџd2
mul_8}
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	d*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    Ш   2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2
strided_slice_2u
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2

MatMul_6q
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
Tanh^
mul_9MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
mul_9_
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
add_3}
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	d*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  2
strided_slice_3/stack
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2
strided_slice_3u
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2

MatMul_7q
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
	Sigmoid_2U
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
Tanh_1d
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
mul_10^
IdentityIdentity
mul_10:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2

Identityb

Identity_1Identity
mul_10:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity_1a

Identity_2Identity	add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:џџџџџџџџџЌ:џџџџџџџџџd:џџџџџџџџџd::::P L
(
_output_shapes
:џџџџџџџџџЌ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџd
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџd
"
_user_specified_name
states/1
І
О
while_cond_31793
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_31793___redundant_placeholder03
/while_while_cond_31793___redundant_placeholder13
/while_while_cond_31793___redundant_placeholder23
/while_while_cond_31793___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџd:џџџџџџџџџd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
:
§

$__inference_lstm_layer_call_fn_34903

inputs
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCallќ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_326672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:џџџџџџџџџ(Ќ:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ(Ќ
 
_user_specified_nameinputs
Њ
Q
5__inference_temporal_mean_pooling_layer_call_fn_34921

inputs
identityЮ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_temporal_mean_pooling_layer_call_and_return_conditional_losses_327052
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџd:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
с

+__inference_dense_final_layer_call_fn_34941

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_final_layer_call_and_return_conditional_losses_327292
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџd::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
№
Щ
E__inference_sequential_layer_call_and_return_conditional_losses_32766
embedding_input
embedding_32749

lstm_32752

lstm_32754

lstm_32756
dense_final_32760
dense_final_32762
identityЂ#dense_final/StatefulPartitionedCallЂ!embedding/StatefulPartitionedCallЂlstm/StatefulPartitionedCall
!embedding/StatefulPartitionedCallStatefulPartitionedCallembedding_inputembedding_32749*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ(Ќ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_320182#
!embedding/StatefulPartitionedCallЏ
lstm/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0
lstm_32752
lstm_32754
lstm_32756*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_326672
lstm/StatefulPartitionedCall
%temporal_mean_pooling/PartitionedCallPartitionedCall%lstm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_temporal_mean_pooling_layer_call_and_return_conditional_losses_327052'
%temporal_mean_pooling/PartitionedCallШ
#dense_final/StatefulPartitionedCallStatefulPartitionedCall.temporal_mean_pooling/PartitionedCall:output:0dense_final_32760dense_final_32762*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_final_layer_call_and_return_conditional_losses_327292%
#dense_final/StatefulPartitionedCallщ
IdentityIdentity,dense_final/StatefulPartitionedCall:output:0$^dense_final/StatefulPartitionedCall"^embedding/StatefulPartitionedCall^lstm/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ(::::::2J
#dense_final/StatefulPartitionedCall#dense_final/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:X T
'
_output_shapes
:џџџџџџџџџ(
)
_user_specified_nameembedding_input
њ
Ф
*__inference_sequential_layer_call_fn_32804
embedding_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityЂStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallembedding_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_327892
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ(::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:џџџџџџџџџ(
)
_user_specified_nameembedding_input
Г
Ў
F__inference_dense_final_layer_call_and_return_conditional_losses_34932

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџd:::O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
њ
Ф
*__inference_sequential_layer_call_fn_32841
embedding_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityЂStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallembedding_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_328262
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ(::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:џџџџџџџџџ(
)
_user_specified_nameembedding_input
Мё
я
?__inference_lstm_layer_call_and_return_conditional_losses_32412

inputs-
)lstm_cell_4_split_readvariableop_resource/
+lstm_cell_4_split_1_readvariableop_resource'
#lstm_cell_4_readvariableop_resource
identityЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:(џџџџџџџџџЌ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2§
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
shrink_axis_mask2
strided_slice_2
lstm_cell_4/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell_4/ones_like/Shape
lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_4/ones_like/ConstЕ
lstm_cell_4/ones_likeFill$lstm_cell_4/ones_like/Shape:output:0$lstm_cell_4/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/ones_like{
lstm_cell_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2
lstm_cell_4/dropout/ConstА
lstm_cell_4/dropout/MulMullstm_cell_4/ones_like:output:0"lstm_cell_4/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/dropout/Mul
lstm_cell_4/dropout/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout/Shapeј
0lstm_cell_4/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_4/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*
seedБџх)*
seed2аЇ22
0lstm_cell_4/dropout/random_uniform/RandomUniform
"lstm_cell_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2$
"lstm_cell_4/dropout/GreaterEqual/yя
 lstm_cell_4/dropout/GreaterEqualGreaterEqual9lstm_cell_4/dropout/random_uniform/RandomUniform:output:0+lstm_cell_4/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2"
 lstm_cell_4/dropout/GreaterEqualЄ
lstm_cell_4/dropout/CastCast$lstm_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/dropout/CastЋ
lstm_cell_4/dropout/Mul_1Mullstm_cell_4/dropout/Mul:z:0lstm_cell_4/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/dropout/Mul_1
lstm_cell_4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2
lstm_cell_4/dropout_1/ConstЖ
lstm_cell_4/dropout_1/MulMullstm_cell_4/ones_like:output:0$lstm_cell_4/dropout_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/dropout_1/Mul
lstm_cell_4/dropout_1/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout_1/Shape§
2lstm_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_1/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*
seedБџх)*
seed2јПY24
2lstm_cell_4/dropout_1/random_uniform/RandomUniform
$lstm_cell_4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2&
$lstm_cell_4/dropout_1/GreaterEqual/yї
"lstm_cell_4/dropout_1/GreaterEqualGreaterEqual;lstm_cell_4/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2$
"lstm_cell_4/dropout_1/GreaterEqualЊ
lstm_cell_4/dropout_1/CastCast&lstm_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/dropout_1/CastГ
lstm_cell_4/dropout_1/Mul_1Mullstm_cell_4/dropout_1/Mul:z:0lstm_cell_4/dropout_1/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/dropout_1/Mul_1
lstm_cell_4/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2
lstm_cell_4/dropout_2/ConstЖ
lstm_cell_4/dropout_2/MulMullstm_cell_4/ones_like:output:0$lstm_cell_4/dropout_2/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/dropout_2/Mul
lstm_cell_4/dropout_2/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout_2/Shapeў
2lstm_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_2/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*
seedБџх)*
seed2п24
2lstm_cell_4/dropout_2/random_uniform/RandomUniform
$lstm_cell_4/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2&
$lstm_cell_4/dropout_2/GreaterEqual/yї
"lstm_cell_4/dropout_2/GreaterEqualGreaterEqual;lstm_cell_4/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2$
"lstm_cell_4/dropout_2/GreaterEqualЊ
lstm_cell_4/dropout_2/CastCast&lstm_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/dropout_2/CastГ
lstm_cell_4/dropout_2/Mul_1Mullstm_cell_4/dropout_2/Mul:z:0lstm_cell_4/dropout_2/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/dropout_2/Mul_1
lstm_cell_4/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2
lstm_cell_4/dropout_3/ConstЖ
lstm_cell_4/dropout_3/MulMullstm_cell_4/ones_like:output:0$lstm_cell_4/dropout_3/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/dropout_3/Mul
lstm_cell_4/dropout_3/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout_3/Shapeў
2lstm_cell_4/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_3/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
dtype0*
seedБџх)*
seed2Д24
2lstm_cell_4/dropout_3/random_uniform/RandomUniform
$lstm_cell_4/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2&
$lstm_cell_4/dropout_3/GreaterEqual/yї
"lstm_cell_4/dropout_3/GreaterEqualGreaterEqual;lstm_cell_4/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2$
"lstm_cell_4/dropout_3/GreaterEqualЊ
lstm_cell_4/dropout_3/CastCast&lstm_cell_4/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/dropout_3/CastГ
lstm_cell_4/dropout_3/Mul_1Mullstm_cell_4/dropout_3/Mul:z:0lstm_cell_4/dropout_3/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/dropout_3/Mul_1|
lstm_cell_4/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_4/ones_like_1/Shape
lstm_cell_4/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_4/ones_like_1/ConstМ
lstm_cell_4/ones_like_1Fill&lstm_cell_4/ones_like_1/Shape:output:0&lstm_cell_4/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/ones_like_1
lstm_cell_4/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2
lstm_cell_4/dropout_4/ConstЗ
lstm_cell_4/dropout_4/MulMul lstm_cell_4/ones_like_1:output:0$lstm_cell_4/dropout_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/dropout_4/Mul
lstm_cell_4/dropout_4/ShapeShape lstm_cell_4/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout_4/Shape§
2lstm_cell_4/dropout_4/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџd*
dtype0*
seedБџх)*
seed2сЕј24
2lstm_cell_4/dropout_4/random_uniform/RandomUniform
$lstm_cell_4/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2&
$lstm_cell_4/dropout_4/GreaterEqual/yі
"lstm_cell_4/dropout_4/GreaterEqualGreaterEqual;lstm_cell_4/dropout_4/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2$
"lstm_cell_4/dropout_4/GreaterEqualЉ
lstm_cell_4/dropout_4/CastCast&lstm_cell_4/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/dropout_4/CastВ
lstm_cell_4/dropout_4/Mul_1Mullstm_cell_4/dropout_4/Mul:z:0lstm_cell_4/dropout_4/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/dropout_4/Mul_1
lstm_cell_4/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2
lstm_cell_4/dropout_5/ConstЗ
lstm_cell_4/dropout_5/MulMul lstm_cell_4/ones_like_1:output:0$lstm_cell_4/dropout_5/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/dropout_5/Mul
lstm_cell_4/dropout_5/ShapeShape lstm_cell_4/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout_5/Shapeќ
2lstm_cell_4/dropout_5/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџd*
dtype0*
seedБџх)*
seed2З24
2lstm_cell_4/dropout_5/random_uniform/RandomUniform
$lstm_cell_4/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2&
$lstm_cell_4/dropout_5/GreaterEqual/yі
"lstm_cell_4/dropout_5/GreaterEqualGreaterEqual;lstm_cell_4/dropout_5/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2$
"lstm_cell_4/dropout_5/GreaterEqualЉ
lstm_cell_4/dropout_5/CastCast&lstm_cell_4/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/dropout_5/CastВ
lstm_cell_4/dropout_5/Mul_1Mullstm_cell_4/dropout_5/Mul:z:0lstm_cell_4/dropout_5/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/dropout_5/Mul_1
lstm_cell_4/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2
lstm_cell_4/dropout_6/ConstЗ
lstm_cell_4/dropout_6/MulMul lstm_cell_4/ones_like_1:output:0$lstm_cell_4/dropout_6/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/dropout_6/Mul
lstm_cell_4/dropout_6/ShapeShape lstm_cell_4/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout_6/Shape§
2lstm_cell_4/dropout_6/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_6/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџd*
dtype0*
seedБџх)*
seed2жяМ24
2lstm_cell_4/dropout_6/random_uniform/RandomUniform
$lstm_cell_4/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2&
$lstm_cell_4/dropout_6/GreaterEqual/yі
"lstm_cell_4/dropout_6/GreaterEqualGreaterEqual;lstm_cell_4/dropout_6/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2$
"lstm_cell_4/dropout_6/GreaterEqualЉ
lstm_cell_4/dropout_6/CastCast&lstm_cell_4/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/dropout_6/CastВ
lstm_cell_4/dropout_6/Mul_1Mullstm_cell_4/dropout_6/Mul:z:0lstm_cell_4/dropout_6/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/dropout_6/Mul_1
lstm_cell_4/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUU@2
lstm_cell_4/dropout_7/ConstЗ
lstm_cell_4/dropout_7/MulMul lstm_cell_4/ones_like_1:output:0$lstm_cell_4/dropout_7/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/dropout_7/Mul
lstm_cell_4/dropout_7/ShapeShape lstm_cell_4/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout_7/Shape§
2lstm_cell_4/dropout_7/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_7/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџd*
dtype0*
seedБџх)*
seed2ї24
2lstm_cell_4/dropout_7/random_uniform/RandomUniform
$lstm_cell_4/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?2&
$lstm_cell_4/dropout_7/GreaterEqual/yі
"lstm_cell_4/dropout_7/GreaterEqualGreaterEqual;lstm_cell_4/dropout_7/random_uniform/RandomUniform:output:0-lstm_cell_4/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2$
"lstm_cell_4/dropout_7/GreaterEqualЉ
lstm_cell_4/dropout_7/CastCast&lstm_cell_4/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/dropout_7/CastВ
lstm_cell_4/dropout_7/Mul_1Mullstm_cell_4/dropout_7/Mul:z:0lstm_cell_4/dropout_7/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/dropout_7/Mul_1
lstm_cell_4/mulMulstrided_slice_2:output:0lstm_cell_4/dropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/mul
lstm_cell_4/mul_1Mulstrided_slice_2:output:0lstm_cell_4/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/mul_1
lstm_cell_4/mul_2Mulstrided_slice_2:output:0lstm_cell_4/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/mul_2
lstm_cell_4/mul_3Mulstrided_slice_2:output:0lstm_cell_4/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/mul_3h
lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/Const|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dimА
 lstm_cell_4/split/ReadVariableOpReadVariableOp)lstm_cell_4_split_readvariableop_resource* 
_output_shapes
:
Ќ*
dtype02"
 lstm_cell_4/split/ReadVariableOpл
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0(lstm_cell_4/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	Ќd:	Ќd:	Ќd:	Ќd*
	num_split2
lstm_cell_4/split
lstm_cell_4/MatMulMatMullstm_cell_4/mul:z:0lstm_cell_4/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul
lstm_cell_4/MatMul_1MatMullstm_cell_4/mul_1:z:0lstm_cell_4/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul_1
lstm_cell_4/MatMul_2MatMullstm_cell_4/mul_2:z:0lstm_cell_4/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul_2
lstm_cell_4/MatMul_3MatMullstm_cell_4/mul_3:z:0lstm_cell_4/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul_3l
lstm_cell_4/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/Const_1
lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_4/split_1/split_dimБ
"lstm_cell_4/split_1/ReadVariableOpReadVariableOp+lstm_cell_4_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02$
"lstm_cell_4/split_1/ReadVariableOpЯ
lstm_cell_4/split_1Split&lstm_cell_4/split_1/split_dim:output:0*lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:d:d:d:d*
	num_split2
lstm_cell_4/split_1Ѓ
lstm_cell_4/BiasAddBiasAddlstm_cell_4/MatMul:product:0lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/BiasAddЉ
lstm_cell_4/BiasAdd_1BiasAddlstm_cell_4/MatMul_1:product:0lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/BiasAdd_1Љ
lstm_cell_4/BiasAdd_2BiasAddlstm_cell_4/MatMul_2:product:0lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/BiasAdd_2Љ
lstm_cell_4/BiasAdd_3BiasAddlstm_cell_4/MatMul_3:product:0lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/BiasAdd_3
lstm_cell_4/mul_4Mulzeros:output:0lstm_cell_4/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/mul_4
lstm_cell_4/mul_5Mulzeros:output:0lstm_cell_4/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/mul_5
lstm_cell_4/mul_6Mulzeros:output:0lstm_cell_4/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/mul_6
lstm_cell_4/mul_7Mulzeros:output:0lstm_cell_4/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/mul_7
lstm_cell_4/ReadVariableOpReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	d*
dtype02
lstm_cell_4/ReadVariableOp
lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell_4/strided_slice/stack
!lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   2#
!lstm_cell_4/strided_slice/stack_1
!lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell_4/strided_slice/stack_2Ф
lstm_cell_4/strided_sliceStridedSlice"lstm_cell_4/ReadVariableOp:value:0(lstm_cell_4/strided_slice/stack:output:0*lstm_cell_4/strided_slice/stack_1:output:0*lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2
lstm_cell_4/strided_sliceЃ
lstm_cell_4/MatMul_4MatMullstm_cell_4/mul_4:z:0"lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul_4
lstm_cell_4/addAddV2lstm_cell_4/BiasAdd:output:0lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/add|
lstm_cell_4/SigmoidSigmoidlstm_cell_4/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/SigmoidЁ
lstm_cell_4/ReadVariableOp_1ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	d*
dtype02
lstm_cell_4/ReadVariableOp_1
!lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   2#
!lstm_cell_4/strided_slice_1/stack
#lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Ш   2%
#lstm_cell_4/strided_slice_1/stack_1
#lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_4/strided_slice_1/stack_2а
lstm_cell_4/strided_slice_1StridedSlice$lstm_cell_4/ReadVariableOp_1:value:0*lstm_cell_4/strided_slice_1/stack:output:0,lstm_cell_4/strided_slice_1/stack_1:output:0,lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2
lstm_cell_4/strided_slice_1Ѕ
lstm_cell_4/MatMul_5MatMullstm_cell_4/mul_5:z:0$lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul_5Ё
lstm_cell_4/add_1AddV2lstm_cell_4/BiasAdd_1:output:0lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/add_1
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/Sigmoid_1
lstm_cell_4/mul_8Mullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/mul_8Ё
lstm_cell_4/ReadVariableOp_2ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	d*
dtype02
lstm_cell_4/ReadVariableOp_2
!lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    Ш   2#
!lstm_cell_4/strided_slice_2/stack
#lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  2%
#lstm_cell_4/strided_slice_2/stack_1
#lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_4/strided_slice_2/stack_2а
lstm_cell_4/strided_slice_2StridedSlice$lstm_cell_4/ReadVariableOp_2:value:0*lstm_cell_4/strided_slice_2/stack:output:0,lstm_cell_4/strided_slice_2/stack_1:output:0,lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2
lstm_cell_4/strided_slice_2Ѕ
lstm_cell_4/MatMul_6MatMullstm_cell_4/mul_6:z:0$lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul_6Ё
lstm_cell_4/add_2AddV2lstm_cell_4/BiasAdd_2:output:0lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/add_2u
lstm_cell_4/TanhTanhlstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/Tanh
lstm_cell_4/mul_9Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/mul_9
lstm_cell_4/add_3AddV2lstm_cell_4/mul_8:z:0lstm_cell_4/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/add_3Ё
lstm_cell_4/ReadVariableOp_3ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	d*
dtype02
lstm_cell_4/ReadVariableOp_3
!lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  2#
!lstm_cell_4/strided_slice_3/stack
#lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_4/strided_slice_3/stack_1
#lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_4/strided_slice_3/stack_2а
lstm_cell_4/strided_slice_3StridedSlice$lstm_cell_4/ReadVariableOp_3:value:0*lstm_cell_4/strided_slice_3/stack:output:0,lstm_cell_4/strided_slice_3/stack_1:output:0,lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2
lstm_cell_4/strided_slice_3Ѕ
lstm_cell_4/MatMul_7MatMullstm_cell_4/mul_7:z:0$lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul_7Ё
lstm_cell_4/add_4AddV2lstm_cell_4/BiasAdd_3:output:0lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/add_4
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Tanh_1Tanhlstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/Tanh_1
lstm_cell_4/mul_10Mullstm_cell_4/Sigmoid_2:y:0lstm_cell_4/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/mul_10
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterп
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_4_split_readvariableop_resource+lstm_cell_4_split_1_readvariableop_resource#lstm_cell_4_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_32212*
condR
while_cond_32211*K
output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:(џџџџџџџџџd*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЅ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ(d2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:џџџџџџџџџ(Ќ:::2
whilewhile:T P
,
_output_shapes
:џџџџџџџџџ(Ќ
 
_user_specified_nameinputs
п
Л
*__inference_sequential_layer_call_fn_33549

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityЂStatefulPartitionedCallЉ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_327892
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ(::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ(
 
_user_specified_nameinputs
е
Р
E__inference_sequential_layer_call_and_return_conditional_losses_32789

inputs
embedding_32772

lstm_32775

lstm_32777

lstm_32779
dense_final_32783
dense_final_32785
identityЂ#dense_final/StatefulPartitionedCallЂ!embedding/StatefulPartitionedCallЂlstm/StatefulPartitionedCall
!embedding/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_32772*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ(Ќ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_320182#
!embedding/StatefulPartitionedCallЏ
lstm/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0
lstm_32775
lstm_32777
lstm_32779*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_324122
lstm/StatefulPartitionedCall
%temporal_mean_pooling/PartitionedCallPartitionedCall%lstm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_temporal_mean_pooling_layer_call_and_return_conditional_losses_327012'
%temporal_mean_pooling/PartitionedCallШ
#dense_final/StatefulPartitionedCallStatefulPartitionedCall.temporal_mean_pooling/PartitionedCall:output:0dense_final_32783dense_final_32785*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_final_layer_call_and_return_conditional_losses_327292%
#dense_final/StatefulPartitionedCallщ
IdentityIdentity,dense_final/StatefulPartitionedCall:output:0$^dense_final/StatefulPartitionedCall"^embedding/StatefulPartitionedCall^lstm/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ(::::::2J
#dense_final/StatefulPartitionedCall#dense_final/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ(
 
_user_specified_nameinputs
бЉ
Џ

 sequential_lstm_while_body_31085<
8sequential_lstm_while_sequential_lstm_while_loop_counterB
>sequential_lstm_while_sequential_lstm_while_maximum_iterations%
!sequential_lstm_while_placeholder'
#sequential_lstm_while_placeholder_1'
#sequential_lstm_while_placeholder_2'
#sequential_lstm_while_placeholder_3;
7sequential_lstm_while_sequential_lstm_strided_slice_1_0w
ssequential_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensor_0E
Asequential_lstm_while_lstm_cell_4_split_readvariableop_resource_0G
Csequential_lstm_while_lstm_cell_4_split_1_readvariableop_resource_0?
;sequential_lstm_while_lstm_cell_4_readvariableop_resource_0"
sequential_lstm_while_identity$
 sequential_lstm_while_identity_1$
 sequential_lstm_while_identity_2$
 sequential_lstm_while_identity_3$
 sequential_lstm_while_identity_4$
 sequential_lstm_while_identity_59
5sequential_lstm_while_sequential_lstm_strided_slice_1u
qsequential_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensorC
?sequential_lstm_while_lstm_cell_4_split_readvariableop_resourceE
Asequential_lstm_while_lstm_cell_4_split_1_readvariableop_resource=
9sequential_lstm_while_lstm_cell_4_readvariableop_resourceу
Gsequential/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  2I
Gsequential/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeД
9sequential/lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemssequential_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensor_0!sequential_lstm_while_placeholderPsequential/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџЌ*
element_dtype02;
9sequential/lstm/while/TensorArrayV2Read/TensorListGetItemж
1sequential/lstm/while/lstm_cell_4/ones_like/ShapeShape@sequential/lstm/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:23
1sequential/lstm/while/lstm_cell_4/ones_like/ShapeЋ
1sequential/lstm/while/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?23
1sequential/lstm/while/lstm_cell_4/ones_like/Const
+sequential/lstm/while/lstm_cell_4/ones_likeFill:sequential/lstm/while/lstm_cell_4/ones_like/Shape:output:0:sequential/lstm/while/lstm_cell_4/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2-
+sequential/lstm/while/lstm_cell_4/ones_likeН
3sequential/lstm/while/lstm_cell_4/ones_like_1/ShapeShape#sequential_lstm_while_placeholder_2*
T0*
_output_shapes
:25
3sequential/lstm/while/lstm_cell_4/ones_like_1/ShapeЏ
3sequential/lstm/while/lstm_cell_4/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?25
3sequential/lstm/while/lstm_cell_4/ones_like_1/Const
-sequential/lstm/while/lstm_cell_4/ones_like_1Fill<sequential/lstm/while/lstm_cell_4/ones_like_1/Shape:output:0<sequential/lstm/while/lstm_cell_4/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2/
-sequential/lstm/while/lstm_cell_4/ones_like_1
%sequential/lstm/while/lstm_cell_4/mulMul@sequential/lstm/while/TensorArrayV2Read/TensorListGetItem:item:04sequential/lstm/while/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2'
%sequential/lstm/while/lstm_cell_4/mul
'sequential/lstm/while/lstm_cell_4/mul_1Mul@sequential/lstm/while/TensorArrayV2Read/TensorListGetItem:item:04sequential/lstm/while/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2)
'sequential/lstm/while/lstm_cell_4/mul_1
'sequential/lstm/while/lstm_cell_4/mul_2Mul@sequential/lstm/while/TensorArrayV2Read/TensorListGetItem:item:04sequential/lstm/while/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2)
'sequential/lstm/while/lstm_cell_4/mul_2
'sequential/lstm/while/lstm_cell_4/mul_3Mul@sequential/lstm/while/TensorArrayV2Read/TensorListGetItem:item:04sequential/lstm/while/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2)
'sequential/lstm/while/lstm_cell_4/mul_3
'sequential/lstm/while/lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2)
'sequential/lstm/while/lstm_cell_4/ConstЈ
1sequential/lstm/while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1sequential/lstm/while/lstm_cell_4/split/split_dimє
6sequential/lstm/while/lstm_cell_4/split/ReadVariableOpReadVariableOpAsequential_lstm_while_lstm_cell_4_split_readvariableop_resource_0* 
_output_shapes
:
Ќ*
dtype028
6sequential/lstm/while/lstm_cell_4/split/ReadVariableOpГ
'sequential/lstm/while/lstm_cell_4/splitSplit:sequential/lstm/while/lstm_cell_4/split/split_dim:output:0>sequential/lstm/while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	Ќd:	Ќd:	Ќd:	Ќd*
	num_split2)
'sequential/lstm/while/lstm_cell_4/splitэ
(sequential/lstm/while/lstm_cell_4/MatMulMatMul)sequential/lstm/while/lstm_cell_4/mul:z:00sequential/lstm/while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2*
(sequential/lstm/while/lstm_cell_4/MatMulѓ
*sequential/lstm/while/lstm_cell_4/MatMul_1MatMul+sequential/lstm/while/lstm_cell_4/mul_1:z:00sequential/lstm/while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd2,
*sequential/lstm/while/lstm_cell_4/MatMul_1ѓ
*sequential/lstm/while/lstm_cell_4/MatMul_2MatMul+sequential/lstm/while/lstm_cell_4/mul_2:z:00sequential/lstm/while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd2,
*sequential/lstm/while/lstm_cell_4/MatMul_2ѓ
*sequential/lstm/while/lstm_cell_4/MatMul_3MatMul+sequential/lstm/while/lstm_cell_4/mul_3:z:00sequential/lstm/while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџd2,
*sequential/lstm/while/lstm_cell_4/MatMul_3
)sequential/lstm/while/lstm_cell_4/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2+
)sequential/lstm/while/lstm_cell_4/Const_1Ќ
3sequential/lstm/while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 25
3sequential/lstm/while/lstm_cell_4/split_1/split_dimѕ
8sequential/lstm/while/lstm_cell_4/split_1/ReadVariableOpReadVariableOpCsequential_lstm_while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02:
8sequential/lstm/while/lstm_cell_4/split_1/ReadVariableOpЇ
)sequential/lstm/while/lstm_cell_4/split_1Split<sequential/lstm/while/lstm_cell_4/split_1/split_dim:output:0@sequential/lstm/while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:d:d:d:d*
	num_split2+
)sequential/lstm/while/lstm_cell_4/split_1ћ
)sequential/lstm/while/lstm_cell_4/BiasAddBiasAdd2sequential/lstm/while/lstm_cell_4/MatMul:product:02sequential/lstm/while/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2+
)sequential/lstm/while/lstm_cell_4/BiasAdd
+sequential/lstm/while/lstm_cell_4/BiasAdd_1BiasAdd4sequential/lstm/while/lstm_cell_4/MatMul_1:product:02sequential/lstm/while/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџd2-
+sequential/lstm/while/lstm_cell_4/BiasAdd_1
+sequential/lstm/while/lstm_cell_4/BiasAdd_2BiasAdd4sequential/lstm/while/lstm_cell_4/MatMul_2:product:02sequential/lstm/while/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџd2-
+sequential/lstm/while/lstm_cell_4/BiasAdd_2
+sequential/lstm/while/lstm_cell_4/BiasAdd_3BiasAdd4sequential/lstm/while/lstm_cell_4/MatMul_3:product:02sequential/lstm/while/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџd2-
+sequential/lstm/while/lstm_cell_4/BiasAdd_3ш
'sequential/lstm/while/lstm_cell_4/mul_4Mul#sequential_lstm_while_placeholder_26sequential/lstm/while/lstm_cell_4/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2)
'sequential/lstm/while/lstm_cell_4/mul_4ш
'sequential/lstm/while/lstm_cell_4/mul_5Mul#sequential_lstm_while_placeholder_26sequential/lstm/while/lstm_cell_4/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2)
'sequential/lstm/while/lstm_cell_4/mul_5ш
'sequential/lstm/while/lstm_cell_4/mul_6Mul#sequential_lstm_while_placeholder_26sequential/lstm/while/lstm_cell_4/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2)
'sequential/lstm/while/lstm_cell_4/mul_6ш
'sequential/lstm/while/lstm_cell_4/mul_7Mul#sequential_lstm_while_placeholder_26sequential/lstm/while/lstm_cell_4/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2)
'sequential/lstm/while/lstm_cell_4/mul_7с
0sequential/lstm/while/lstm_cell_4/ReadVariableOpReadVariableOp;sequential_lstm_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	d*
dtype022
0sequential/lstm/while/lstm_cell_4/ReadVariableOpП
5sequential/lstm/while/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5sequential/lstm/while/lstm_cell_4/strided_slice/stackУ
7sequential/lstm/while/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   29
7sequential/lstm/while/lstm_cell_4/strided_slice/stack_1У
7sequential/lstm/while/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7sequential/lstm/while/lstm_cell_4/strided_slice/stack_2Ш
/sequential/lstm/while/lstm_cell_4/strided_sliceStridedSlice8sequential/lstm/while/lstm_cell_4/ReadVariableOp:value:0>sequential/lstm/while/lstm_cell_4/strided_slice/stack:output:0@sequential/lstm/while/lstm_cell_4/strided_slice/stack_1:output:0@sequential/lstm/while/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask21
/sequential/lstm/while/lstm_cell_4/strided_sliceћ
*sequential/lstm/while/lstm_cell_4/MatMul_4MatMul+sequential/lstm/while/lstm_cell_4/mul_4:z:08sequential/lstm/while/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2,
*sequential/lstm/while/lstm_cell_4/MatMul_4ѓ
%sequential/lstm/while/lstm_cell_4/addAddV22sequential/lstm/while/lstm_cell_4/BiasAdd:output:04sequential/lstm/while/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2'
%sequential/lstm/while/lstm_cell_4/addО
)sequential/lstm/while/lstm_cell_4/SigmoidSigmoid)sequential/lstm/while/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2+
)sequential/lstm/while/lstm_cell_4/Sigmoidх
2sequential/lstm/while/lstm_cell_4/ReadVariableOp_1ReadVariableOp;sequential_lstm_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	d*
dtype024
2sequential/lstm/while/lstm_cell_4/ReadVariableOp_1У
7sequential/lstm/while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   29
7sequential/lstm/while/lstm_cell_4/strided_slice_1/stackЧ
9sequential/lstm/while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Ш   2;
9sequential/lstm/while/lstm_cell_4/strided_slice_1/stack_1Ч
9sequential/lstm/while/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9sequential/lstm/while/lstm_cell_4/strided_slice_1/stack_2д
1sequential/lstm/while/lstm_cell_4/strided_slice_1StridedSlice:sequential/lstm/while/lstm_cell_4/ReadVariableOp_1:value:0@sequential/lstm/while/lstm_cell_4/strided_slice_1/stack:output:0Bsequential/lstm/while/lstm_cell_4/strided_slice_1/stack_1:output:0Bsequential/lstm/while/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask23
1sequential/lstm/while/lstm_cell_4/strided_slice_1§
*sequential/lstm/while/lstm_cell_4/MatMul_5MatMul+sequential/lstm/while/lstm_cell_4/mul_5:z:0:sequential/lstm/while/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2,
*sequential/lstm/while/lstm_cell_4/MatMul_5љ
'sequential/lstm/while/lstm_cell_4/add_1AddV24sequential/lstm/while/lstm_cell_4/BiasAdd_1:output:04sequential/lstm/while/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2)
'sequential/lstm/while/lstm_cell_4/add_1Ф
+sequential/lstm/while/lstm_cell_4/Sigmoid_1Sigmoid+sequential/lstm/while/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2-
+sequential/lstm/while/lstm_cell_4/Sigmoid_1с
'sequential/lstm/while/lstm_cell_4/mul_8Mul/sequential/lstm/while/lstm_cell_4/Sigmoid_1:y:0#sequential_lstm_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџd2)
'sequential/lstm/while/lstm_cell_4/mul_8х
2sequential/lstm/while/lstm_cell_4/ReadVariableOp_2ReadVariableOp;sequential_lstm_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	d*
dtype024
2sequential/lstm/while/lstm_cell_4/ReadVariableOp_2У
7sequential/lstm/while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    Ш   29
7sequential/lstm/while/lstm_cell_4/strided_slice_2/stackЧ
9sequential/lstm/while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  2;
9sequential/lstm/while/lstm_cell_4/strided_slice_2/stack_1Ч
9sequential/lstm/while/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9sequential/lstm/while/lstm_cell_4/strided_slice_2/stack_2д
1sequential/lstm/while/lstm_cell_4/strided_slice_2StridedSlice:sequential/lstm/while/lstm_cell_4/ReadVariableOp_2:value:0@sequential/lstm/while/lstm_cell_4/strided_slice_2/stack:output:0Bsequential/lstm/while/lstm_cell_4/strided_slice_2/stack_1:output:0Bsequential/lstm/while/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask23
1sequential/lstm/while/lstm_cell_4/strided_slice_2§
*sequential/lstm/while/lstm_cell_4/MatMul_6MatMul+sequential/lstm/while/lstm_cell_4/mul_6:z:0:sequential/lstm/while/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2,
*sequential/lstm/while/lstm_cell_4/MatMul_6љ
'sequential/lstm/while/lstm_cell_4/add_2AddV24sequential/lstm/while/lstm_cell_4/BiasAdd_2:output:04sequential/lstm/while/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2)
'sequential/lstm/while/lstm_cell_4/add_2З
&sequential/lstm/while/lstm_cell_4/TanhTanh+sequential/lstm/while/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2(
&sequential/lstm/while/lstm_cell_4/Tanhц
'sequential/lstm/while/lstm_cell_4/mul_9Mul-sequential/lstm/while/lstm_cell_4/Sigmoid:y:0*sequential/lstm/while/lstm_cell_4/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2)
'sequential/lstm/while/lstm_cell_4/mul_9ч
'sequential/lstm/while/lstm_cell_4/add_3AddV2+sequential/lstm/while/lstm_cell_4/mul_8:z:0+sequential/lstm/while/lstm_cell_4/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2)
'sequential/lstm/while/lstm_cell_4/add_3х
2sequential/lstm/while/lstm_cell_4/ReadVariableOp_3ReadVariableOp;sequential_lstm_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	d*
dtype024
2sequential/lstm/while/lstm_cell_4/ReadVariableOp_3У
7sequential/lstm/while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  29
7sequential/lstm/while/lstm_cell_4/strided_slice_3/stackЧ
9sequential/lstm/while/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9sequential/lstm/while/lstm_cell_4/strided_slice_3/stack_1Ч
9sequential/lstm/while/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9sequential/lstm/while/lstm_cell_4/strided_slice_3/stack_2д
1sequential/lstm/while/lstm_cell_4/strided_slice_3StridedSlice:sequential/lstm/while/lstm_cell_4/ReadVariableOp_3:value:0@sequential/lstm/while/lstm_cell_4/strided_slice_3/stack:output:0Bsequential/lstm/while/lstm_cell_4/strided_slice_3/stack_1:output:0Bsequential/lstm/while/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask23
1sequential/lstm/while/lstm_cell_4/strided_slice_3§
*sequential/lstm/while/lstm_cell_4/MatMul_7MatMul+sequential/lstm/while/lstm_cell_4/mul_7:z:0:sequential/lstm/while/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2,
*sequential/lstm/while/lstm_cell_4/MatMul_7љ
'sequential/lstm/while/lstm_cell_4/add_4AddV24sequential/lstm/while/lstm_cell_4/BiasAdd_3:output:04sequential/lstm/while/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2)
'sequential/lstm/while/lstm_cell_4/add_4Ф
+sequential/lstm/while/lstm_cell_4/Sigmoid_2Sigmoid+sequential/lstm/while/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2-
+sequential/lstm/while/lstm_cell_4/Sigmoid_2Л
(sequential/lstm/while/lstm_cell_4/Tanh_1Tanh+sequential/lstm/while/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2*
(sequential/lstm/while/lstm_cell_4/Tanh_1ь
(sequential/lstm/while/lstm_cell_4/mul_10Mul/sequential/lstm/while/lstm_cell_4/Sigmoid_2:y:0,sequential/lstm/while/lstm_cell_4/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2*
(sequential/lstm/while/lstm_cell_4/mul_10А
:sequential/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#sequential_lstm_while_placeholder_1!sequential_lstm_while_placeholder,sequential/lstm/while/lstm_cell_4/mul_10:z:0*
_output_shapes
: *
element_dtype02<
:sequential/lstm/while/TensorArrayV2Write/TensorListSetItem|
sequential/lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
sequential/lstm/while/add/yЉ
sequential/lstm/while/addAddV2!sequential_lstm_while_placeholder$sequential/lstm/while/add/y:output:0*
T0*
_output_shapes
: 2
sequential/lstm/while/add
sequential/lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
sequential/lstm/while/add_1/yЦ
sequential/lstm/while/add_1AddV28sequential_lstm_while_sequential_lstm_while_loop_counter&sequential/lstm/while/add_1/y:output:0*
T0*
_output_shapes
: 2
sequential/lstm/while/add_1
sequential/lstm/while/IdentityIdentitysequential/lstm/while/add_1:z:0*
T0*
_output_shapes
: 2 
sequential/lstm/while/IdentityБ
 sequential/lstm/while/Identity_1Identity>sequential_lstm_while_sequential_lstm_while_maximum_iterations*
T0*
_output_shapes
: 2"
 sequential/lstm/while/Identity_1
 sequential/lstm/while/Identity_2Identitysequential/lstm/while/add:z:0*
T0*
_output_shapes
: 2"
 sequential/lstm/while/Identity_2Н
 sequential/lstm/while/Identity_3IdentityJsequential/lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2"
 sequential/lstm/while/Identity_3А
 sequential/lstm/while/Identity_4Identity,sequential/lstm/while/lstm_cell_4/mul_10:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2"
 sequential/lstm/while/Identity_4Џ
 sequential/lstm/while/Identity_5Identity+sequential/lstm/while/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2"
 sequential/lstm/while/Identity_5"I
sequential_lstm_while_identity'sequential/lstm/while/Identity:output:0"M
 sequential_lstm_while_identity_1)sequential/lstm/while/Identity_1:output:0"M
 sequential_lstm_while_identity_2)sequential/lstm/while/Identity_2:output:0"M
 sequential_lstm_while_identity_3)sequential/lstm/while/Identity_3:output:0"M
 sequential_lstm_while_identity_4)sequential/lstm/while/Identity_4:output:0"M
 sequential_lstm_while_identity_5)sequential/lstm/while/Identity_5:output:0"x
9sequential_lstm_while_lstm_cell_4_readvariableop_resource;sequential_lstm_while_lstm_cell_4_readvariableop_resource_0"
Asequential_lstm_while_lstm_cell_4_split_1_readvariableop_resourceCsequential_lstm_while_lstm_cell_4_split_1_readvariableop_resource_0"
?sequential_lstm_while_lstm_cell_4_split_readvariableop_resourceAsequential_lstm_while_lstm_cell_4_split_readvariableop_resource_0"p
5sequential_lstm_while_sequential_lstm_strided_slice_17sequential_lstm_while_sequential_lstm_strided_slice_1_0"ш
qsequential_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensorssequential_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџd:џџџџџџџџџd: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
: 
т
l
P__inference_temporal_mean_pooling_layer_call_and_return_conditional_losses_32705

inputs
identityZ
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџd:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
Й
Ы
+__inference_lstm_cell_4_layer_call_fn_35190

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2ЂStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_314162
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:џџџџџџџџџЌ:џџџџџџџџџd:џџџџџџџџџd:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџЌ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџd
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџd
"
_user_specified_name
states/1
Й
Ы
+__inference_lstm_cell_4_layer_call_fn_35207

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2ЂStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_315002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:џџџџџџџџџЌ:џџџџџџџџџd:џџџџџџџџџd:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџЌ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџd
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџd
"
_user_specified_name
states/1
№
ё
?__inference_lstm_layer_call_and_return_conditional_losses_34221
inputs_0-
)lstm_cell_4_split_readvariableop_resource/
+lstm_cell_4_split_1_readvariableop_resource'
#lstm_cell_4_readvariableop_resource
identityЂwhileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2§
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
shrink_axis_mask2
strided_slice_2
lstm_cell_4/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell_4/ones_like/Shape
lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_4/ones_like/ConstЕ
lstm_cell_4/ones_likeFill$lstm_cell_4/ones_like/Shape:output:0$lstm_cell_4/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/ones_like|
lstm_cell_4/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_4/ones_like_1/Shape
lstm_cell_4/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_4/ones_like_1/ConstМ
lstm_cell_4/ones_like_1Fill&lstm_cell_4/ones_like_1/Shape:output:0&lstm_cell_4/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/ones_like_1
lstm_cell_4/mulMulstrided_slice_2:output:0lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/mul
lstm_cell_4/mul_1Mulstrided_slice_2:output:0lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/mul_1
lstm_cell_4/mul_2Mulstrided_slice_2:output:0lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/mul_2
lstm_cell_4/mul_3Mulstrided_slice_2:output:0lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_4/mul_3h
lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/Const|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dimА
 lstm_cell_4/split/ReadVariableOpReadVariableOp)lstm_cell_4_split_readvariableop_resource* 
_output_shapes
:
Ќ*
dtype02"
 lstm_cell_4/split/ReadVariableOpл
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0(lstm_cell_4/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	Ќd:	Ќd:	Ќd:	Ќd*
	num_split2
lstm_cell_4/split
lstm_cell_4/MatMulMatMullstm_cell_4/mul:z:0lstm_cell_4/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul
lstm_cell_4/MatMul_1MatMullstm_cell_4/mul_1:z:0lstm_cell_4/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul_1
lstm_cell_4/MatMul_2MatMullstm_cell_4/mul_2:z:0lstm_cell_4/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul_2
lstm_cell_4/MatMul_3MatMullstm_cell_4/mul_3:z:0lstm_cell_4/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul_3l
lstm_cell_4/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/Const_1
lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_4/split_1/split_dimБ
"lstm_cell_4/split_1/ReadVariableOpReadVariableOp+lstm_cell_4_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02$
"lstm_cell_4/split_1/ReadVariableOpЯ
lstm_cell_4/split_1Split&lstm_cell_4/split_1/split_dim:output:0*lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:d:d:d:d*
	num_split2
lstm_cell_4/split_1Ѓ
lstm_cell_4/BiasAddBiasAddlstm_cell_4/MatMul:product:0lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/BiasAddЉ
lstm_cell_4/BiasAdd_1BiasAddlstm_cell_4/MatMul_1:product:0lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/BiasAdd_1Љ
lstm_cell_4/BiasAdd_2BiasAddlstm_cell_4/MatMul_2:product:0lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/BiasAdd_2Љ
lstm_cell_4/BiasAdd_3BiasAddlstm_cell_4/MatMul_3:product:0lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/BiasAdd_3
lstm_cell_4/mul_4Mulzeros:output:0 lstm_cell_4/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/mul_4
lstm_cell_4/mul_5Mulzeros:output:0 lstm_cell_4/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/mul_5
lstm_cell_4/mul_6Mulzeros:output:0 lstm_cell_4/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/mul_6
lstm_cell_4/mul_7Mulzeros:output:0 lstm_cell_4/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/mul_7
lstm_cell_4/ReadVariableOpReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	d*
dtype02
lstm_cell_4/ReadVariableOp
lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell_4/strided_slice/stack
!lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   2#
!lstm_cell_4/strided_slice/stack_1
!lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell_4/strided_slice/stack_2Ф
lstm_cell_4/strided_sliceStridedSlice"lstm_cell_4/ReadVariableOp:value:0(lstm_cell_4/strided_slice/stack:output:0*lstm_cell_4/strided_slice/stack_1:output:0*lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2
lstm_cell_4/strided_sliceЃ
lstm_cell_4/MatMul_4MatMullstm_cell_4/mul_4:z:0"lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul_4
lstm_cell_4/addAddV2lstm_cell_4/BiasAdd:output:0lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/add|
lstm_cell_4/SigmoidSigmoidlstm_cell_4/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/SigmoidЁ
lstm_cell_4/ReadVariableOp_1ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	d*
dtype02
lstm_cell_4/ReadVariableOp_1
!lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   2#
!lstm_cell_4/strided_slice_1/stack
#lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Ш   2%
#lstm_cell_4/strided_slice_1/stack_1
#lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_4/strided_slice_1/stack_2а
lstm_cell_4/strided_slice_1StridedSlice$lstm_cell_4/ReadVariableOp_1:value:0*lstm_cell_4/strided_slice_1/stack:output:0,lstm_cell_4/strided_slice_1/stack_1:output:0,lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2
lstm_cell_4/strided_slice_1Ѕ
lstm_cell_4/MatMul_5MatMullstm_cell_4/mul_5:z:0$lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul_5Ё
lstm_cell_4/add_1AddV2lstm_cell_4/BiasAdd_1:output:0lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/add_1
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/Sigmoid_1
lstm_cell_4/mul_8Mullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/mul_8Ё
lstm_cell_4/ReadVariableOp_2ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	d*
dtype02
lstm_cell_4/ReadVariableOp_2
!lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    Ш   2#
!lstm_cell_4/strided_slice_2/stack
#lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  2%
#lstm_cell_4/strided_slice_2/stack_1
#lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_4/strided_slice_2/stack_2а
lstm_cell_4/strided_slice_2StridedSlice$lstm_cell_4/ReadVariableOp_2:value:0*lstm_cell_4/strided_slice_2/stack:output:0,lstm_cell_4/strided_slice_2/stack_1:output:0,lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2
lstm_cell_4/strided_slice_2Ѕ
lstm_cell_4/MatMul_6MatMullstm_cell_4/mul_6:z:0$lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul_6Ё
lstm_cell_4/add_2AddV2lstm_cell_4/BiasAdd_2:output:0lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/add_2u
lstm_cell_4/TanhTanhlstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/Tanh
lstm_cell_4/mul_9Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/mul_9
lstm_cell_4/add_3AddV2lstm_cell_4/mul_8:z:0lstm_cell_4/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/add_3Ё
lstm_cell_4/ReadVariableOp_3ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	d*
dtype02
lstm_cell_4/ReadVariableOp_3
!lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  2#
!lstm_cell_4/strided_slice_3/stack
#lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_4/strided_slice_3/stack_1
#lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_4/strided_slice_3/stack_2а
lstm_cell_4/strided_slice_3StridedSlice$lstm_cell_4/ReadVariableOp_3:value:0*lstm_cell_4/strided_slice_3/stack:output:0,lstm_cell_4/strided_slice_3/stack_1:output:0,lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2
lstm_cell_4/strided_slice_3Ѕ
lstm_cell_4/MatMul_7MatMullstm_cell_4/mul_7:z:0$lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/MatMul_7Ё
lstm_cell_4/add_4AddV2lstm_cell_4/BiasAdd_3:output:0lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/add_4
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Tanh_1Tanhlstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/Tanh_1
lstm_cell_4/mul_10Mullstm_cell_4/Sigmoid_2:y:0lstm_cell_4/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
lstm_cell_4/mul_10
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterп
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_4_split_readvariableop_resource+lstm_cell_4_split_1_readvariableop_resource#lstm_cell_4_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_34085*
condR
while_cond_34084*K
output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџd*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџd2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:џџџџџџџџџџџџџџџџџџЌ:::2
whilewhile:_ [
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ
"
_user_specified_name
inputs/0
э$

while_body_31926
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_4_31950_0
while_lstm_cell_4_31952_0
while_lstm_cell_4_31954_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_4_31950
while_lstm_cell_4_31952
while_lstm_cell_4_31954Ђ)while/lstm_cell_4/StatefulPartitionedCallУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeд
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџЌ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemз
)while/lstm_cell_4/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_4_31950_0while_lstm_cell_4_31952_0while_lstm_cell_4_31954_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_315002+
)while/lstm_cell_4/StatefulPartitionedCallі
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_4/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2Й
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Т
while/Identity_4Identity2while/lstm_cell_4/StatefulPartitionedCall:output:1*^while/lstm_cell_4/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџd2
while/Identity_4Т
while/Identity_5Identity2while/lstm_cell_4/StatefulPartitionedCall:output:2*^while/lstm_cell_4/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџd2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_4_31950while_lstm_cell_4_31950_0"4
while_lstm_cell_4_31952while_lstm_cell_4_31952_0"4
while_lstm_cell_4_31954while_lstm_cell_4_31954_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџd:џџџџџџџџџd: : :::2V
)while/lstm_cell_4/StatefulPartitionedCall)while/lstm_cell_4/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
: 
ПD
б
?__inference_lstm_layer_call_and_return_conditional_losses_31995

inputs
lstm_cell_4_31913
lstm_cell_4_31915
lstm_cell_4_31917
identityЂ#lstm_cell_4/StatefulPartitionedCallЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2§
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџЌ*
shrink_axis_mask2
strided_slice_2
#lstm_cell_4/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_4_31913lstm_cell_4_31915lstm_cell_4_31917*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџd:џџџџџџџџџd:џџџџџџџџџd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_315002%
#lstm_cell_4/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_4_31913lstm_cell_4_31915lstm_cell_4_31917*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_31926*
condR
while_cond_31925*K
output_shapes:
8: : : : :џџџџџџџџџd:џџџџџџџџџd: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџd*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџd*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџd2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_4/StatefulPartitionedCall^while*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:џџџџџџџџџџџџџџџџџџЌ:::2J
#lstm_cell_4/StatefulPartitionedCall#lstm_cell_4/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџЌ
 
_user_specified_nameinputs
т
l
P__inference_temporal_mean_pooling_layer_call_and_return_conditional_losses_32701

inputs
identityZ
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџd:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
Т[
Я
!__inference__traced_restore_35366
file_prefix)
%assignvariableop_embedding_embeddings)
%assignvariableop_1_dense_final_kernel'
#assignvariableop_2_dense_final_bias#
assignvariableop_3_rmsprop_iter$
 assignvariableop_4_rmsprop_decay,
(assignvariableop_5_rmsprop_learning_rate'
#assignvariableop_6_rmsprop_momentum"
assignvariableop_7_rmsprop_rho.
*assignvariableop_8_lstm_lstm_cell_4_kernel8
4assignvariableop_9_lstm_lstm_cell_4_recurrent_kernel-
)assignvariableop_10_lstm_lstm_cell_4_bias
assignvariableop_11_total
assignvariableop_12_count
assignvariableop_13_total_1
assignvariableop_14_count_18
4assignvariableop_15_rmsprop_embedding_embeddings_rms6
2assignvariableop_16_rmsprop_dense_final_kernel_rms4
0assignvariableop_17_rmsprop_dense_final_bias_rms;
7assignvariableop_18_rmsprop_lstm_lstm_cell_4_kernel_rmsE
Aassignvariableop_19_rmsprop_lstm_lstm_cell_4_recurrent_kernel_rms9
5assignvariableop_20_rmsprop_lstm_lstm_cell_4_bias_rms
identity_22ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ѓ

value
B
B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesК
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*l
_output_shapesZ
X::::::::::::::::::::::*$
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЄ
AssignVariableOpAssignVariableOp%assignvariableop_embedding_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Њ
AssignVariableOp_1AssignVariableOp%assignvariableop_1_dense_final_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ј
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_final_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_3Є
AssignVariableOp_3AssignVariableOpassignvariableop_3_rmsprop_iterIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Ѕ
AssignVariableOp_4AssignVariableOp assignvariableop_4_rmsprop_decayIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5­
AssignVariableOp_5AssignVariableOp(assignvariableop_5_rmsprop_learning_rateIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Ј
AssignVariableOp_6AssignVariableOp#assignvariableop_6_rmsprop_momentumIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ѓ
AssignVariableOp_7AssignVariableOpassignvariableop_7_rmsprop_rhoIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Џ
AssignVariableOp_8AssignVariableOp*assignvariableop_8_lstm_lstm_cell_4_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Й
AssignVariableOp_9AssignVariableOp4assignvariableop_9_lstm_lstm_cell_4_recurrent_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Б
AssignVariableOp_10AssignVariableOp)assignvariableop_10_lstm_lstm_cell_4_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ё
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ё
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Ѓ
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ѓ
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15М
AssignVariableOp_15AssignVariableOp4assignvariableop_15_rmsprop_embedding_embeddings_rmsIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16К
AssignVariableOp_16AssignVariableOp2assignvariableop_16_rmsprop_dense_final_kernel_rmsIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17И
AssignVariableOp_17AssignVariableOp0assignvariableop_17_rmsprop_dense_final_bias_rmsIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18П
AssignVariableOp_18AssignVariableOp7assignvariableop_18_rmsprop_lstm_lstm_cell_4_kernel_rmsIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Щ
AssignVariableOp_19AssignVariableOpAassignvariableop_19_rmsprop_lstm_lstm_cell_4_recurrent_kernel_rmsIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Н
AssignVariableOp_20AssignVariableOp5assignvariableop_20_rmsprop_lstm_lstm_cell_4_bias_rmsIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_209
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpЌ
Identity_21Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_21
Identity_22IdentityIdentity_21:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_22"#
identity_22Identity_22:output:0*i
_input_shapesX
V: :::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Њ
Q
5__inference_temporal_mean_pooling_layer_call_fn_34916

inputs
identityЮ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_temporal_mean_pooling_layer_call_and_return_conditional_losses_327012
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџd:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
ж
п
while_body_32531
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_05
1while_lstm_cell_4_split_readvariableop_resource_07
3while_lstm_cell_4_split_1_readvariableop_resource_0/
+while_lstm_cell_4_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor3
/while_lstm_cell_4_split_readvariableop_resource5
1while_lstm_cell_4_split_1_readvariableop_resource-
)while_lstm_cell_4_readvariableop_resourceУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ,  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeд
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџЌ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemІ
!while/lstm_cell_4/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/ones_like/Shape
!while/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!while/lstm_cell_4/ones_like/ConstЭ
while/lstm_cell_4/ones_likeFill*while/lstm_cell_4/ones_like/Shape:output:0*while/lstm_cell_4/ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_4/ones_like
#while/lstm_cell_4/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2%
#while/lstm_cell_4/ones_like_1/Shape
#while/lstm_cell_4/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2%
#while/lstm_cell_4/ones_like_1/Constд
while/lstm_cell_4/ones_like_1Fill,while/lstm_cell_4/ones_like_1/Shape:output:0,while/lstm_cell_4/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/ones_like_1Р
while/lstm_cell_4/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_4/mulФ
while/lstm_cell_4/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_4/mul_1Ф
while/lstm_cell_4/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_4/mul_2Ф
while/lstm_cell_4/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_4/ones_like:output:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_4/mul_3t
while/lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_4/Const
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_4/split/split_dimФ
&while/lstm_cell_4/split/ReadVariableOpReadVariableOp1while_lstm_cell_4_split_readvariableop_resource_0* 
_output_shapes
:
Ќ*
dtype02(
&while/lstm_cell_4/split/ReadVariableOpѓ
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0.while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	Ќd:	Ќd:	Ќd:	Ќd*
	num_split2
while/lstm_cell_4/split­
while/lstm_cell_4/MatMulMatMulwhile/lstm_cell_4/mul:z:0 while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMulГ
while/lstm_cell_4/MatMul_1MatMulwhile/lstm_cell_4/mul_1:z:0 while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMul_1Г
while/lstm_cell_4/MatMul_2MatMulwhile/lstm_cell_4/mul_2:z:0 while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMul_2Г
while/lstm_cell_4/MatMul_3MatMulwhile/lstm_cell_4/mul_3:z:0 while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMul_3x
while/lstm_cell_4/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_4/Const_1
#while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_4/split_1/split_dimХ
(while/lstm_cell_4/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02*
(while/lstm_cell_4/split_1/ReadVariableOpч
while/lstm_cell_4/split_1Split,while/lstm_cell_4/split_1/split_dim:output:00while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:d:d:d:d*
	num_split2
while/lstm_cell_4/split_1Л
while/lstm_cell_4/BiasAddBiasAdd"while/lstm_cell_4/MatMul:product:0"while/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/BiasAddС
while/lstm_cell_4/BiasAdd_1BiasAdd$while/lstm_cell_4/MatMul_1:product:0"while/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/BiasAdd_1С
while/lstm_cell_4/BiasAdd_2BiasAdd$while/lstm_cell_4/MatMul_2:product:0"while/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/BiasAdd_2С
while/lstm_cell_4/BiasAdd_3BiasAdd$while/lstm_cell_4/MatMul_3:product:0"while/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/BiasAdd_3Ј
while/lstm_cell_4/mul_4Mulwhile_placeholder_2&while/lstm_cell_4/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/mul_4Ј
while/lstm_cell_4/mul_5Mulwhile_placeholder_2&while/lstm_cell_4/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/mul_5Ј
while/lstm_cell_4/mul_6Mulwhile_placeholder_2&while/lstm_cell_4/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/mul_6Ј
while/lstm_cell_4/mul_7Mulwhile_placeholder_2&while/lstm_cell_4/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/mul_7Б
 while/lstm_cell_4/ReadVariableOpReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	d*
dtype02"
 while/lstm_cell_4/ReadVariableOp
%while/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell_4/strided_slice/stackЃ
'while/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   2)
'while/lstm_cell_4/strided_slice/stack_1Ѓ
'while/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell_4/strided_slice/stack_2ш
while/lstm_cell_4/strided_sliceStridedSlice(while/lstm_cell_4/ReadVariableOp:value:0.while/lstm_cell_4/strided_slice/stack:output:00while/lstm_cell_4/strided_slice/stack_1:output:00while/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2!
while/lstm_cell_4/strided_sliceЛ
while/lstm_cell_4/MatMul_4MatMulwhile/lstm_cell_4/mul_4:z:0(while/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMul_4Г
while/lstm_cell_4/addAddV2"while/lstm_cell_4/BiasAdd:output:0$while/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/add
while/lstm_cell_4/SigmoidSigmoidwhile/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/SigmoidЕ
"while/lstm_cell_4/ReadVariableOp_1ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	d*
dtype02$
"while/lstm_cell_4/ReadVariableOp_1Ѓ
'while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   2)
'while/lstm_cell_4/strided_slice_1/stackЇ
)while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Ш   2+
)while/lstm_cell_4/strided_slice_1/stack_1Ї
)while/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_4/strided_slice_1/stack_2є
!while/lstm_cell_4/strided_slice_1StridedSlice*while/lstm_cell_4/ReadVariableOp_1:value:00while/lstm_cell_4/strided_slice_1/stack:output:02while/lstm_cell_4/strided_slice_1/stack_1:output:02while/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_1Н
while/lstm_cell_4/MatMul_5MatMulwhile/lstm_cell_4/mul_5:z:0*while/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMul_5Й
while/lstm_cell_4/add_1AddV2$while/lstm_cell_4/BiasAdd_1:output:0$while/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/add_1
while/lstm_cell_4/Sigmoid_1Sigmoidwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/Sigmoid_1Ё
while/lstm_cell_4/mul_8Mulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/mul_8Е
"while/lstm_cell_4/ReadVariableOp_2ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	d*
dtype02$
"while/lstm_cell_4/ReadVariableOp_2Ѓ
'while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    Ш   2)
'while/lstm_cell_4/strided_slice_2/stackЇ
)while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  2+
)while/lstm_cell_4/strided_slice_2/stack_1Ї
)while/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_4/strided_slice_2/stack_2є
!while/lstm_cell_4/strided_slice_2StridedSlice*while/lstm_cell_4/ReadVariableOp_2:value:00while/lstm_cell_4/strided_slice_2/stack:output:02while/lstm_cell_4/strided_slice_2/stack_1:output:02while/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_2Н
while/lstm_cell_4/MatMul_6MatMulwhile/lstm_cell_4/mul_6:z:0*while/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMul_6Й
while/lstm_cell_4/add_2AddV2$while/lstm_cell_4/BiasAdd_2:output:0$while/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/add_2
while/lstm_cell_4/TanhTanhwhile/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/TanhІ
while/lstm_cell_4/mul_9Mulwhile/lstm_cell_4/Sigmoid:y:0while/lstm_cell_4/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/mul_9Ї
while/lstm_cell_4/add_3AddV2while/lstm_cell_4/mul_8:z:0while/lstm_cell_4/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/add_3Е
"while/lstm_cell_4/ReadVariableOp_3ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	d*
dtype02$
"while/lstm_cell_4/ReadVariableOp_3Ѓ
'while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  2)
'while/lstm_cell_4/strided_slice_3/stackЇ
)while/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_4/strided_slice_3/stack_1Ї
)while/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_4/strided_slice_3/stack_2є
!while/lstm_cell_4/strided_slice_3StridedSlice*while/lstm_cell_4/ReadVariableOp_3:value:00while/lstm_cell_4/strided_slice_3/stack:output:02while/lstm_cell_4/strided_slice_3/stack_1:output:02while/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_3Н
while/lstm_cell_4/MatMul_7MatMulwhile/lstm_cell_4/mul_7:z:0*while/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/MatMul_7Й
while/lstm_cell_4/add_4AddV2$while/lstm_cell_4/BiasAdd_3:output:0$while/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/add_4
while/lstm_cell_4/Sigmoid_2Sigmoidwhile/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/Sigmoid_2
while/lstm_cell_4/Tanh_1Tanhwhile/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/Tanh_1Ќ
while/lstm_cell_4/mul_10Mulwhile/lstm_cell_4/Sigmoid_2:y:0while/lstm_cell_4/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/lstm_cell_4/mul_10р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_4/mul_10:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_4/mul_10:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџd2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_4_readvariableop_resource+while_lstm_cell_4_readvariableop_resource_0"h
1while_lstm_cell_4_split_1_readvariableop_resource3while_lstm_cell_4_split_1_readvariableop_resource_0"d
/while_lstm_cell_4_split_readvariableop_resource1while_lstm_cell_4_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџd:џџџџџџџџџd: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџd:-)
'
_output_shapes
:џџџџџџџџџd:

_output_shapes
: :

_output_shapes
: "ИL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*О
serving_defaultЊ
K
embedding_input8
!serving_default_embedding_input:0џџџџџџџџџ(?
dense_final0
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:яЦ
ќ(
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
	optimizer
regularization_losses
trainable_variables
	variables
		keras_api


signatures
\__call__
]_default_save_signature
*^&call_and_return_all_conditional_losses"Џ&
_tf_keras_sequential&{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 40]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "embedding_input"}}, {"class_name": "Embedding", "config": {"name": "embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 40]}, "dtype": "float32", "input_dim": 18476, "output_dim": 300, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 40}}, {"class_name": "LSTM", "config": {"name": "lstm", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 100, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.7, "recurrent_dropout": 0.7, "implementation": 1}}, {"class_name": "TemporalMeanPooling", "config": {"name": "temporal_mean_pooling", "trainable": true, "dtype": "float32"}}, {"class_name": "Dense", "config": {"name": "dense_final", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 40]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "embedding_input"}}, {"class_name": "Embedding", "config": {"name": "embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 40]}, "dtype": "float32", "input_dim": 18476, "output_dim": 300, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 40}}, {"class_name": "LSTM", "config": {"name": "lstm", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 100, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.7, "recurrent_dropout": 0.7, "implementation": 1}}, {"class_name": "TemporalMeanPooling", "config": {"name": "temporal_mean_pooling", "trainable": true, "dtype": "float32"}}, {"class_name": "Dense", "config": {"name": "dense_final", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "binary_crossentropy", "metrics": ["acc"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.0010000000474974513, "decay": 0.0, "rho": 0.8999999761581421, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
І

embeddings
regularization_losses
trainable_variables
	variables
	keras_api
___call__
*`&call_and_return_all_conditional_losses"
_tf_keras_layerэ{"class_name": "Embedding", "name": "embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 40]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 40]}, "dtype": "float32", "input_dim": 18476, "output_dim": 300, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 40}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40]}}
М
cell

state_spec
regularization_losses
trainable_variables
	variables
	keras_api
a__call__
*b&call_and_return_all_conditional_losses"

_tf_keras_rnn_layerѕ	{"class_name": "LSTM", "name": "lstm", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 100, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.7, "recurrent_dropout": 0.7, "implementation": 1}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 300]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 40, 300]}}
й
regularization_losses
trainable_variables
	variables
	keras_api
c__call__
*d&call_and_return_all_conditional_losses"Ъ
_tf_keras_layerА{"class_name": "TemporalMeanPooling", "name": "temporal_mean_pooling", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "temporal_mean_pooling", "trainable": true, "dtype": "float32"}}
ќ

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
e__call__
*f&call_and_return_all_conditional_losses"з
_tf_keras_layerН{"class_name": "Dense", "name": "dense_final", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_final", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}

 iter
	!decay
"learning_rate
#momentum
$rho	rmsV	rmsW	rmsX	%rmsY	&rmsZ	'rms["
	optimizer
 "
trackable_list_wrapper
J
0
%1
&2
'3
4
5"
trackable_list_wrapper
J
0
%1
&2
'3
4
5"
trackable_list_wrapper
Ъ
regularization_losses
(metrics
)layer_regularization_losses
*non_trainable_variables

+layers
trainable_variables
	variables
,layer_metrics
\__call__
]_default_save_signature
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
,
gserving_default"
signature_map
):'ЌЌ2embedding/embeddings
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
­
regularization_losses
-metrics
.layer_regularization_losses
/non_trainable_variables

0layers
trainable_variables
	variables
1layer_metrics
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
Љ

%kernel
&recurrent_kernel
'bias
2regularization_losses
3trainable_variables
4	variables
5	keras_api
h__call__
*i&call_and_return_all_conditional_losses"ю
_tf_keras_layerд{"class_name": "LSTMCell", "name": "lstm_cell_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_4", "trainable": true, "dtype": "float32", "units": 100, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.7, "recurrent_dropout": 0.7, "implementation": 1}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
%0
&1
'2"
trackable_list_wrapper
5
%0
&1
'2"
trackable_list_wrapper
Й

6states
regularization_losses
7metrics
8layer_regularization_losses
9non_trainable_variables

:layers
trainable_variables
	variables
;layer_metrics
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
regularization_losses
<metrics
=layer_regularization_losses
>non_trainable_variables

?layers
trainable_variables
	variables
@layer_metrics
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
$:"d2dense_final/kernel
:2dense_final/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
regularization_losses
Ametrics
Blayer_regularization_losses
Cnon_trainable_variables

Dlayers
trainable_variables
	variables
Elayer_metrics
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
+:)
Ќ2lstm/lstm_cell_4/kernel
4:2	d2!lstm/lstm_cell_4/recurrent_kernel
$:"2lstm/lstm_cell_4/bias
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
%0
&1
'2"
trackable_list_wrapper
5
%0
&1
'2"
trackable_list_wrapper
­
2regularization_losses
Hmetrics
Ilayer_regularization_losses
Jnon_trainable_variables

Klayers
3trainable_variables
4	variables
Llayer_metrics
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Л
	Mtotal
	Ncount
O	variables
P	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
ѕ
	Qtotal
	Rcount
S
_fn_kwargs
T	variables
U	keras_api"Ў
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "acc", "dtype": "float32", "config": {"name": "acc", "dtype": "float32", "fn": "categorical_accuracy"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
.
M0
N1"
trackable_list_wrapper
-
O	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
Q0
R1"
trackable_list_wrapper
-
T	variables"
_generic_user_object
3:1ЌЌ2 RMSprop/embedding/embeddings/rms
.:,d2RMSprop/dense_final/kernel/rms
(:&2RMSprop/dense_final/bias/rms
5:3
Ќ2#RMSprop/lstm/lstm_cell_4/kernel/rms
>:<	d2-RMSprop/lstm/lstm_cell_4/recurrent_kernel/rms
.:,2!RMSprop/lstm/lstm_cell_4/bias/rms
і2ѓ
*__inference_sequential_layer_call_fn_32804
*__inference_sequential_layer_call_fn_32841
*__inference_sequential_layer_call_fn_33566
*__inference_sequential_layer_call_fn_33549Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ц2у
 __inference__wrapped_model_31228О
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *.Ђ+
)&
embedding_inputџџџџџџџџџ(
т2п
E__inference_sequential_layer_call_and_return_conditional_losses_33264
E__inference_sequential_layer_call_and_return_conditional_losses_33532
E__inference_sequential_layer_call_and_return_conditional_losses_32766
E__inference_sequential_layer_call_and_return_conditional_losses_32746Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
г2а
)__inference_embedding_layer_call_fn_33583Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_embedding_layer_call_and_return_conditional_losses_33576Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓ2№
$__inference_lstm_layer_call_fn_34243
$__inference_lstm_layer_call_fn_34892
$__inference_lstm_layer_call_fn_34232
$__inference_lstm_layer_call_fn_34903е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
п2м
?__inference_lstm_layer_call_and_return_conditional_losses_34221
?__inference_lstm_layer_call_and_return_conditional_losses_33966
?__inference_lstm_layer_call_and_return_conditional_losses_34881
?__inference_lstm_layer_call_and_return_conditional_losses_34626е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
К2З
5__inference_temporal_mean_pooling_layer_call_fn_34921
5__inference_temporal_mean_pooling_layer_call_fn_34916Ц
НВЙ
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
№2э
P__inference_temporal_mean_pooling_layer_call_and_return_conditional_losses_34911
P__inference_temporal_mean_pooling_layer_call_and_return_conditional_losses_34907Ц
НВЙ
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
е2в
+__inference_dense_final_layer_call_fn_34941Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№2э
F__inference_dense_final_layer_call_and_return_conditional_losses_34932Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
:B8
#__inference_signature_wrapper_32868embedding_input
2
+__inference_lstm_cell_4_layer_call_fn_35190
+__inference_lstm_cell_4_layer_call_fn_35207О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
д2б
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_35089
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_35173О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 Ё
 __inference__wrapped_model_31228}%'&8Ђ5
.Ђ+
)&
embedding_inputџџџџџџџџџ(
Њ "9Њ6
4
dense_final%"
dense_finalџџџџџџџџџІ
F__inference_dense_final_layer_call_and_return_conditional_losses_34932\/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "%Ђ"

0џџџџџџџџџ
 ~
+__inference_dense_final_layer_call_fn_34941O/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "џџџџџџџџџЈ
D__inference_embedding_layer_call_and_return_conditional_losses_33576`/Ђ,
%Ђ"
 
inputsџџџџџџџџџ(
Њ "*Ђ'
 
0џџџџџџџџџ(Ќ
 
)__inference_embedding_layer_call_fn_33583S/Ђ,
%Ђ"
 
inputsџџџџџџџџџ(
Њ "џџџџџџџџџ(ЌЩ
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_35089ў%'&Ђ~
wЂt
!
inputsџџџџџџџџџЌ
KЂH
"
states/0џџџџџџџџџd
"
states/1џџџџџџџџџd
p
Њ "sЂp
iЂf

0/0џџџџџџџџџd
EB

0/1/0џџџџџџџџџd

0/1/1џџџџџџџџџd
 Щ
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_35173ў%'&Ђ~
wЂt
!
inputsџџџџџџџџџЌ
KЂH
"
states/0џџџџџџџџџd
"
states/1џџџџџџџџџd
p 
Њ "sЂp
iЂf

0/0џџџџџџџџџd
EB

0/1/0џџџџџџџџџd

0/1/1џџџџџџџџџd
 
+__inference_lstm_cell_4_layer_call_fn_35190ю%'&Ђ~
wЂt
!
inputsџџџџџџџџџЌ
KЂH
"
states/0џџџџџџџџџd
"
states/1џџџџџџџџџd
p
Њ "cЂ`

0џџџџџџџџџd
A>

1/0џџџџџџџџџd

1/1џџџџџџџџџd
+__inference_lstm_cell_4_layer_call_fn_35207ю%'&Ђ~
wЂt
!
inputsџџџџџџџџџЌ
KЂH
"
states/0џџџџџџџџџd
"
states/1џџџџџџџџџd
p 
Њ "cЂ`

0џџџџџџџџџd
A>

1/0џџџџџџџџџd

1/1џџџџџџџџџdС
?__inference_lstm_layer_call_and_return_conditional_losses_33966~%'&PЂM
FЂC
52
0-
inputs/0џџџџџџџџџџџџџџџџџџЌ

 
p

 
Њ "%Ђ"

0џџџџџџџџџd
 С
?__inference_lstm_layer_call_and_return_conditional_losses_34221~%'&PЂM
FЂC
52
0-
inputs/0џџџџџџџџџџџџџџџџџџЌ

 
p 

 
Њ "%Ђ"

0џџџџџџџџџd
 Б
?__inference_lstm_layer_call_and_return_conditional_losses_34626n%'&@Ђ=
6Ђ3
%"
inputsџџџџџџџџџ(Ќ

 
p

 
Њ "%Ђ"

0џџџџџџџџџd
 Б
?__inference_lstm_layer_call_and_return_conditional_losses_34881n%'&@Ђ=
6Ђ3
%"
inputsџџџџџџџџџ(Ќ

 
p 

 
Њ "%Ђ"

0џџџџџџџџџd
 
$__inference_lstm_layer_call_fn_34232q%'&PЂM
FЂC
52
0-
inputs/0џџџџџџџџџџџџџџџџџџЌ

 
p

 
Њ "џџџџџџџџџd
$__inference_lstm_layer_call_fn_34243q%'&PЂM
FЂC
52
0-
inputs/0џџџџџџџџџџџџџџџџџџЌ

 
p 

 
Њ "џџџџџџџџџd
$__inference_lstm_layer_call_fn_34892a%'&@Ђ=
6Ђ3
%"
inputsџџџџџџџџџ(Ќ

 
p

 
Њ "џџџџџџџџџd
$__inference_lstm_layer_call_fn_34903a%'&@Ђ=
6Ђ3
%"
inputsџџџџџџџџџ(Ќ

 
p 

 
Њ "џџџџџџџџџdК
E__inference_sequential_layer_call_and_return_conditional_losses_32746q%'&@Ђ=
6Ђ3
)&
embedding_inputџџџџџџџџџ(
p

 
Њ "%Ђ"

0џџџџџџџџџ
 К
E__inference_sequential_layer_call_and_return_conditional_losses_32766q%'&@Ђ=
6Ђ3
)&
embedding_inputџџџџџџџџџ(
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Б
E__inference_sequential_layer_call_and_return_conditional_losses_33264h%'&7Ђ4
-Ђ*
 
inputsџџџџџџџџџ(
p

 
Њ "%Ђ"

0џџџџџџџџџ
 Б
E__inference_sequential_layer_call_and_return_conditional_losses_33532h%'&7Ђ4
-Ђ*
 
inputsџџџџџџџџџ(
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 
*__inference_sequential_layer_call_fn_32804d%'&@Ђ=
6Ђ3
)&
embedding_inputџџџџџџџџџ(
p

 
Њ "џџџџџџџџџ
*__inference_sequential_layer_call_fn_32841d%'&@Ђ=
6Ђ3
)&
embedding_inputџџџџџџџџџ(
p 

 
Њ "џџџџџџџџџ
*__inference_sequential_layer_call_fn_33549[%'&7Ђ4
-Ђ*
 
inputsџџџџџџџџџ(
p

 
Њ "џџџџџџџџџ
*__inference_sequential_layer_call_fn_33566[%'&7Ђ4
-Ђ*
 
inputsџџџџџџџџџ(
p 

 
Њ "џџџџџџџџџИ
#__inference_signature_wrapper_32868%'&KЂH
Ђ 
AЊ>
<
embedding_input)&
embedding_inputџџџџџџџџџ("9Њ6
4
dense_final%"
dense_finalџџџџџџџџџМ
P__inference_temporal_mean_pooling_layer_call_and_return_conditional_losses_34907h?Ђ<
%Ђ"
 
inputsџџџџџџџџџd
Њ

trainingp"%Ђ"

0џџџџџџџџџd
 М
P__inference_temporal_mean_pooling_layer_call_and_return_conditional_losses_34911h?Ђ<
%Ђ"
 
inputsџџџџџџџџџd
Њ

trainingp "%Ђ"

0џџџџџџџџџd
 
5__inference_temporal_mean_pooling_layer_call_fn_34916[?Ђ<
%Ђ"
 
inputsџџџџџџџџџd
Њ

trainingp"џџџџџџџџџd
5__inference_temporal_mean_pooling_layer_call_fn_34921[?Ђ<
%Ђ"
 
inputsџџџџџџџџџd
Њ

trainingp "џџџџџџџџџd