import numpy as np
import MINST_Read as MR
import paddle.v2 as paddle
import matplotlib.pyplot as plt 



paddle.init(use_gpu=False, trainer_count=1)
image_dim = 28*28
class_dim = 10

def network_config():
    image = paddle.layer.data(name='image', type=paddle.data_type.dense_vector(image_dim))
    fc_1 = paddle.layer.fc(input = image,size = 784,act = paddle.activation.Sigmoid())
    fc_2 = paddle.layer.fc(input = fc_1,size = 256,act = paddle.activation.Sigmoid())
    fc_3 = paddle.layer.fc(input = fc_2,size = 64,act = paddle.activation.Sigmoid())
    y_predict = paddle.layer.fc(input = fc_3, size = class_dim, act = paddle.activation.Softmax())
    return  y_predict

y_predict = network_config()
y_label = paddle.layer.data(name='label', type=paddle.data_type.integer_value(class_dim))
cost = paddle.layer.classification_cost(input=y_predict,label=y_label)
parameters = paddle.parameters.create(cost)
optimizer = paddle.optimizer.Momentum(momentum=0.9,regularization = paddle.optimizer.L2Regularization(rate = 0.0002 * 128),
learning_rate = 0.1/128.0,
learning_rate_decay_a = 0.1,
learning_rate_decay_b = 50000 * 100,
learning_rate_schedule = 'discexp')
feeding = {'image': 0,'label': 1}
	
trainer = paddle.trainer.SGD(cost=cost, parameters=parameters, update_equation=optimizer)	


def create_reader(filename,n):
	def reader():
		if filename =='train':
			dataset = MR.fetch_traingset()
		else:
			dataset = MR.fetch_testingset()
		for i in range(n):
			yield dataset['images'][i],dataset['labels'][i]
	return reader
	
def event_handler(event):
    if isinstance(event, paddle.event.EndIteration):
        if event.batch_id % 100 == 0:
            print "Pass %d, Batch %d, Cost %f, %s" % (event.pass_id, event.batch_id, event.cost,event.metrics)
		
            with open('output/params_pass_%d.tar' % event.pass_id, 'w') as para_f:
                 parameters.to_tar(para_f)
            
model_path = 'output/params_pass_0.tar'
def main():
	feeding = {
        'image': 0,
        'label': 1}
	train_reader = create_reader('train',60000)
	trainer.train(reader=paddle.batch(reader = train_reader,batch_size=128),feeding=feeding,event_handler=event_handler,num_passes=1)
	
main()
			


		
