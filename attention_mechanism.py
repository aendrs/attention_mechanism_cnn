# -*- coding: utf-8 -*-
"""
Created on Nov 05 2018
@author: Andrés Méndez  andresmendez@outlook.com

Implementation of an attention mechanism for Convolutional Neural Networks, largely based on the paper: 
[Rodríguez, Pau, et al. "Attend and rectify: a gated attention mechanism for fine-grained recovery." Proceedings of the European Conference on Computer Vision (ECCV). 2018.]

"""

import numpy as np
import tensorflow as tf
from keras.layers import (
    Input,
    BatchNormalization,
    Activation, Dense, Dropout,
    Convolution2D, Conv2D, 
    MaxPooling2D, ZeroPadding2D, multiply,
    GlobalAveragePooling2D, GlobalAveragePooling1D,
    Reshape, concatenate, Lambda, add, Permute,
    SpatialDropout2D
)
from keras.models import Model
from keras.engine import Layer
#from keras.utils.visualize_util import plot
from keras import backend as K



'''
# according to the old paper, new one defines it differently
def attention_head(nheads=1, in_featmaps=1):
    """
     # nheads : number of attention heads
    """
    def f(z):        
        x=Conv2D(nheads,(3,3),padding='same')(z) #convolution to get attention feature maps 
        x=Activation('softmax')(x)
        xRep=K.repeat_elements(x,rep=z.shape[3],axis=3) #repeat the tensor N times along the channel dimension in order to do element-wise multiplication with the feature maps
        zRep=K.repeat_elements(z,rep=nheads,axis=3)
        out=multiply([xRep,zRep]) #element wise multiplication of broadcasted tenbsors, now of the same shape [b, height, width, channels1*channels2]
        return out
    return f
    
    
def reg_featuremaps(weight_matrix):
    [batch,height,width, channels]=weight_matrix.shape
    regLoss=K.cast_to_floatx(0.)
    for b in range(batch):
        for i in range(channels):
            for j in range(channels): 
               W_j_transp= K.transpose(weight_matrix[b,:,:,j])
               if j != i: 
                    tempLoss=K.sqrt(K.square(weight_matrix[b,:,:,i]*W_j_transp))
                    regLoss=regLoss+tempLoss
    return regLoss                
'''  
    

#%%
# regularization loss for the featuremaps 
def reg_featuremaps(featuremaps):  
    def reg_insidebatch(lastoutput,current): #function to be called by tf.scan, which will loop through the batch dimension
        regLoss=K.cast_to_floatx(0.)
        [height, width, channels]=K.int_shape(current)
        for i in range(channels):
            Wi=K.flatten(current[:,:,i])
            for j in range(channels): 
                Wj=K.flatten(current[:,:,j])
                if j != i: 
                    tempLoss=K.sqrt(K.square(K.sum(Wi * Wj,axis=-1,keepdims=False))) 
                    regLoss+=tempLoss
        regLoss+=lastoutput # add the output from the current 3D tensor to the preceding batch loop           
        return regLoss            
    regLoss_total=tf.scan(reg_insidebatch, featuremaps, initializer=0.)            
    return regLoss_total[-1]* 0.1  #since the output is a list choose the last element which correspond to the total sum over the batch dimension              


def attention_head(Kheads,kernel=(3,3), spatialdropoutrate=0.2):
    """
        Kheads : number of attention heads
        z=input = tensor of feature maps, [b, height, width, channels]
    output:
        H_l = Kheads attention heads, [b, height, width, Kheads]
    """
    def f(z):        
        #H_l=Conv2D(Kheads,kernel,padding='same', activation='softmax', activity_regularizer=reg_featuremaps)(z) #convolution to get attention feature maps 
        H_l=Conv2D(Kheads,kernel,padding='same')(z) #convolution to get attention feature maps dim=[b,h,w,channels]
        mapsHeight=H_l._shape_as_list()[1]
        mapsWidth=H_l._shape_as_list()[2]
        H_l=Reshape((mapsHeight*mapsWidth,Kheads))(H_l) #rehape to [b,h*w,channels]        
        H_l=Permute((2,1))(H_l)
        H_l=Activation('softmax')(H_l)
        H_l=Permute((2,1))(H_l) #back to the original dimension order
        H_l=Reshape((mapsWidth,mapsWidth,Kheads))(H_l)       
        H_l=SpatialDropout2D(rate=spatialdropoutrate)(H_l)
        return H_l # [b, height, width, Kheads]
    return f
    
    
def output_head(H_l, Kheads, kernel=(3,3),classes=5):
    """
         Kheads : number of attention heads
         z=input = tensor of feature maps [b, height, width, channels]
         H_l = output from attention_head [b, height, width, Kheads]
         classes: number of classes in the classification model 
     output:
         Tensor of predictions, dim=[b, Kheads, classes]
    """
    def f(z):        
        O_l=Conv2D(Kheads*classes,kernel,padding='same')(z) # (K*classes) output feature maps
        #H_l2=K.repeat_elements(H_l,classes,-1) #repeat featuremaps elements-times before doing element-wise multiplication
        H_l2=Lambda(K.repeat_elements,arguments={'rep':classes,'axis':-1})(H_l)
        o_l=multiply([O_l, H_l2])
        o_l=GlobalAveragePooling2D()(o_l)
        o_l=Reshape((Kheads,classes))(o_l)
        return o_l # output [batch,K,classes]
    return f
    

def attention_gate(H_l, Kheads, kernel=(3,3), classes=5):
    """
        H_l = output from attention_head [b, height, width, Kheads]
        Kheads : number of attention heads
        kernel: kernel size for the conovolution operation
    output:
        gH_l: gate value used to weigh the diverse output heads, dim=[b,Kheads]
    """
    def f(z):
        WgZl=Conv2D(Kheads,kernel,padding='same')(z)
        WgZlHl=multiply([WgZl, H_l])
        gH_l=GlobalAveragePooling2D()(WgZlHl)
        gH_l=Activation('tanh')(gH_l)
        gH_l=Activation('softmax')(gH_l)
        return gH_l
    return f

def layer_gating(gH_l,o_l,classes=5):
    """
    The actual operation of weighing and reducing the Output of the K attention heads into a single prediction vector
        gH_l: gate value used to weigh the diverse output heads, output of the attention_gate, dim=[b,Kheads]
        o_l:  tensor of predictions, output of the output_head, dim=[b, Kheads, classes]
    output
        out_l: output vector dim=[b,classes]
    """
    #gH=K.expand_dims(gH_l,axis=-1)
    gH=Lambda(expand_dims,arguments={'axis':-1})(gH_l)
    #gH=K.repeat_elements(gH,classes,-1)
    gH=Lambda(K.repeat_elements,arguments={'rep':classes,'axis':-1})(gH)
    out_l=multiply([gH,o_l])
    #out_l=K.permute_dimensions(out_l,(0,2,1))
    out_l=GlobalAveragePooling1D()(out_l) #since it is 1D pooling, the average is done over the 1st dimension K [b,K,classes]
    return out_l #dim=[b,classes]


def global_gates(AttModules, O_attmodules, O_network, gatecase=1, classes=5, kernel=(1,1), O_network_weight=0.5, spatialdropoutrate=0.2): 
    """
    AttModules: number of attention modules in the model
    z : feature maps from the last convolutional layer, dim: [b,height,width,channels]
    O_attmodules = tensor with the concatenated output of the diverse attention modules, dim=[batch,AttModules,classes]
    O_network = tensor with the network output, dim=[batch,classes]
    kernel = kernel size for the internal convolutions in the attention_head and layer_gating
    gatecase = selects the type of gating and weighting of the network output and the attention modules output
            gatecase=0 : outputs only the output from the network, (O_network), i.e., without the contribution from the Attention Modules
            gatecase=1 : O_network is concatenated to O_attmodules and weighted with them as equal
            gatecase=2 :  O_network is weighted by a factor O_net_weight and summed to the 
                            automatic weighting of attention modules 
            gatecase=3 : O_network is weighted by a factor O_net_weight and summed to the average of O_attmodules (like in the paper)                
    """
    def f(z):
        if gatecase==0: # Only the output from the network, (O_network), i.e., without the contribution from the Attention Modules
            O_final=O_network
        elif gatecase==1: # O_network is concatenated to O_attmodules and weighted with them as equal         
            #O_network_expand=K.expand_dims(O_network, axis=1)# dim=[b,1,classes]
            O_network_expand=Lambda(expand_dims)(O_network)
            O_concatenated =concatenate([O_network_expand, O_attmodules], axis=1)   #dim=[b, AttModules+1, classes]
            AttHead=attention_head(AttModules+1,kernel, spatialdropoutrate)(z) #AttHead dim=[b, height, width, AttModules+1]
            AttGate=attention_gate(AttHead, AttModules+1, kernel, classes)(z) #AttGate dim=[b, AttModules+1]
            O_final=layer_gating(AttGate,O_concatenated,classes) #O_final dim=[b,classes]
        elif gatecase==2: # O_network is weighted by a factor O_net_weight and summed to the automatic weighting of attention modules 
            AttHead=attention_head(AttModules,kernel,spatialdropoutrate)(z) #AttHead dim=[b, height, width, AttModules]
            AttGate=attention_gate(AttHead, AttModules, kernel, classes)(z) #AttGate dim=[b, AttModules]
            O_attmodules_weighted=layer_gating(AttGate,O_attmodules,classes) # dim=[b,classes]
            #---
            O_network_weighted=Lambda(lambda x: x*O_network_weight)(O_network) # O_network weighted by a factor
            #---
            O_final=add([O_attmodules_weighted,O_network_weighted])
            O_final=Activation('softmax')(O_final) #softmax used in order to normalize the output, maybe it's not needed
            
        elif gatecase==3: # O_network is weighted by a factor O_net_weight and summed to the average of O_attmodules (like in the paper)
            O_attmodules_weighted=Lambda(tensor_mean)(O_attmodules)
            #*******
            O_attmodules_weighted=Lambda(lambda x: x*(1-O_network_weight))(O_attmodules_weighted) #weight O_attmodules by factor 1-O_network_weight
            #*******
            O_network_weighted=Lambda(lambda x: x*O_network_weight)(O_network) # O_network weighted by a factor             
            O_final=add([O_attmodules_weighted,O_network_weighted])
            O_final=Activation('softmax')(O_final) #softmax used in order to normalize the output, maybe it's not needed?                  
        else:
            raise Exception('Invalid value for argument GATECASE')
        return O_final
    return f




def AttentionModule(Kheads,kernel=(3,3),classes=5,add_dim=1,spatialdropoutrate=0.2):
    #add_dim=flag to indicate if we need to add a dummy dimension in order to have out_l dim=[b,1,classes] instead of dim=[b,classes]
    def f(z):
        H_l=attention_head(Kheads,kernel,spatialdropoutrate)(z)
        o_l=output_head(H_l, Kheads, kernel, classes)(z)
        gH_l=attention_gate(H_l, Kheads, kernel, classes)(z)
        out_l=layer_gating(gH_l,o_l,classes)  
        if add_dim:
            #out_l=K.expand_dims(out_l, axis=1)   
            out_l=Lambda(expand_dims)(out_l) #add a dummy dimension in order to have out_l dim=[b,1,classes] 
        return out_l    
    return f    
        

def expand_dims(x,axis=1):
    return K.expand_dims(x,axis)

def tensor_mean(x,axis=1):
    return K.mean(x, axis)

#%%
"""


# TEST ZONEEEE ========================================================
    
Kheads=6
classes=5
z=K.random_uniform_variable(shape=(2,4,4,3), low=0, high=1)


y=AttentionModule(Kheads,kernel=(3,3),classes=5,add_dim=0)(z)


H_l=attention_head(Kheads=Kheads)(z)
o_l=output_head(H_l=H_l, Kheads=Kheads, classes=classes)(z)
gH_l=attention_gate(H_l=H_l,Kheads=Kheads,classes=classes)(z)
out_l=layer_gating(gH_l=gH_l,o_l=o_l,classes=classes)









Ototaltest=K.random_uniform_variable(shape=(2,8,5), low=0, high=1) #8 attmodules

Oouttest=global_gates(8, Ototaltest, classes=5)(z)


def testx(O_network_w=0.5): 
    def f(z):
        y=Lambda(lambda x: x*O_network_weight)(z)
        return y
    return f
    
y=testx(O_network_w=0.3)(z)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    #print(z2.eval())
    print(z2.output_shape)
    #print(a.eval())



z=K.random_uniform_variable(shape=(2,5), low=0, high=1)
z1=K.random_uniform_variable(shape=(2,5), low=0, high=1)

z2=concatenate([z,z1])


zm=K.mean(z,axis=1)

#%%

def expand_dims(x):
    return K.expand_dims(x, axis=1)

y2=Lambda(expand_dims)(y)

y2=K.expand_dims(y,axis=1)

y2=K.repeat_elements(y,5,-1)


conv_atthead=Conv2D(2,(3,3),padding='same')(block1)
out_atthead=Activation('softmax')(conv_atthead)

Wi=K.flatten(out_atthead[:,:,1])
blockA=attention_head(nheads=2)(block1)

out=Reshape((3,5))(out)



from numpy import array
from numpy import tensordot
A = array([1,2])
B = array([3,4])
C = tensordot(A, B, axes=0)
print(C)




def regularization_test(featuremaps):  
    def reg_insidebatch(lastoutput,current): #function to be called by tf.scan, which will loop through the batch dimension
        regLoss=K.cast_to_floatx(0.)
        [height, width, channels]=K.int_shape(current)
        for i in range(channels):
            Wi=K.flatten(current[:,:,i])
            
            tempLoss=K.sum(Wi,axis=-1,keepdims=False)
            regLoss+=tempLoss
            '''
            for j in range(channels): 
                Wj=K.flatten(current[:,:,j])
                if j != i: 
                    tempLoss=K.sqrt(K.square(K.sum(Wi * Wj,axis=-1,keepdims=False))) 
                    regLoss+=tempLoss
            '''        
        print('lastoutput.shape {}'.format(lastoutput.shape))
        print('curent shape {}'.format(K.int_shape(current)))
        print('Wi shape {} ___'.format(Wi.shape))
                    
        regLoss+=lastoutput # add the output from the current 3D tensor to the preceding batch loop           
        return regLoss             
    
    regLoss_total=tf.scan(reg_insidebatch, featuremaps, initializer=K.cast_to_floatx(0.))            
    return regLoss_total[-1]     



#try the regularization with a sample tensor in order to check the result
a=K.random_uniform_variable(shape=(3,4,4,3), low=0, high=1)
b=K.random_uniform_variable(shape=(3,4,4,3), low=0, high=1)
c=multiply([a,b])
d=GlobalAveragePooling2D()(c)

b=reg_featuremaps(a)


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print(b.eval())
    print(b.shape)
    #print(a.eval())



Wi=K.flatten(K.random_uniform_variable(shape=(4,3), low=1, high=1))
Wj=K.flatten(K.random_uniform_variable(shape=(4,3), low=1, high=1))
b=K.sqrt(K.square(K.sum(Wi * Wj,axis=-1,keepdims=False))) 

b=K.sum(K.abs(a), axis=-1)










 Wtransposed= K.transpose(z[0,:,:,1:10])
    
def fractal_block(join_gen,c, filter, nb_col, nb_row, drop_p, convname, dropout=None):
    def f(z):
        columns = [[z] for _ in range(c)]
        last_row = 2**(c-1) - 1
        blockcounter=0
        for row in range(2**(c-1)):
            t_row = []
            for col in range(c):
                prop = 2**(col)
                convnameX=convname+'_Col'+str(col)+'_Row'+str(row)+'_'+str(blockcounter) #compose the name for the convolution block
                # Add blocks
                if (row+1) % prop == 0:
                    t_col = columns[col]
                    t_col.append(fractal_conv(filter=filter,
                                              nb_col=nb_col,
                                              nb_row=nb_row,
                                              dropout=dropout,
                                              convname=convnameX)(t_col[-1]))
                    t_row.append(col)
                    blockcounter += 1
            # Merge (if needed)
            if len(t_row) > 1:
                merging = [columns[x][-1] for x in t_row]
                merged  = join_gen.get_join_layer(drop_p=drop_p)(merging)
                for i in t_row:
                    columns[i].append(merged)
        return columns[0][-1]
    return f    



"""