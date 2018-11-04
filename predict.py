import numpy as np
import MINST_Read as MR
import paddle.v2 as paddle
import matplotlib.pyplot as plt 

image_dim = 28*28
class_dim = 10
paddle.init(use_gpu=False, trainer_count=1)

def network_config():
	image = paddle.layer.data(name='image', type=paddle.data_type.dense_vector(image_dim))
	fc_1 = paddle.layer.fc(input = image,size = 784,act = paddle.activation.Sigmoid())
	fc_2 = paddle.layer.fc(input = fc_1,size = 256,act = paddle.activation.Sigmoid())
	fc_3 = paddle.layer.fc(input = fc_2,size = 64,act = paddle.activation.Sigmoid())
	y_predict = paddle.layer.fc(input = fc_3, size = class_dim, act = paddle.activation.Softmax())
	return  y_predict


number = input('Input the index you want to visit from the database ')
data = MR.fetch_testingset()
image_1 = data['images'][int(number)]
print('label : %d' % data['labels'][int(number)])
image_3 = np.reshape(image_1,[28,28])
image_2 = np.reshape(image_1,[1,784])
plt.imshow(image_3,cmap = 'gray')
plt.show()
model_path = 'output/params_pass_44.tar'

output_layer = network_config()
with open(model_path,'r') as openfile:
	parameters = paddle.parameters.Parameters.from_tar(openfile)
	print parameters
	result = paddle.infer(input = [image_2],parameters = parameters,output_layer=output_layer,feeding = {'image':0})
	print result
	prob = np.max(result)
	index = np.argsort(result)
	print 'The image is %d'%index[0][-1],'and the probabilty is %f'%prob 
