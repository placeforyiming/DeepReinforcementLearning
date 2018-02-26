

from Preprocess import StateProcessor

from Q_function import Q_function
import tensorflow as tf
import numpy as np
from os import path
from os import remove
import json
import random
import gym

# set parameters

model_method='DQN'

N_Q=939         #how many times the Q has been updated.(continue to train from)
         		#how many episodes to play the game

N_frame=4  		 #take how many frame as input
N_delay=4        #how long the action indure


#max_want_train_step=10000
max_train_step=10000  # how many steps the model train
Max_N_Q=1000
batch_size=32
gamma=0.99           # delay factor
initial_num_in_episode=50000

env = gym.make('Pong-v0')

lr=0.00025
lr_eve=0.00025


#Build policy network
with tf.name_scope('main_network'):
	Q=Q_function(batch_size=batch_size,n_frame=N_frame,action_num=6)
	Q_value=Q.CNN_part()

with tf.name_scope('target_network'):
	Target_Q=Q_function(batch_size=batch_size,n_frame=N_frame,action_num=6)
	Target_Q_value=Target_Q.CNN_part()

main_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES , scope='main_network') 
target_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES , scope='target_network')
assign_ops = []
for main_var, target_var in zip(main_variables, target_variables):
	assign_ops.append(tf.assign(target_var, tf.identity(main_var)))
copy_operation = tf.group(*assign_ops)
Act=tf.placeholder(tf.int32,shape=(batch_size,))
Rew=tf.placeholder(tf.float32,shape=(batch_size,))
Q_value_perAct=[]
for i in range(batch_size):
	Q_value_perAct.append(Q_value[i,Act[i]])
Q_value_perAct=tf.transpose(tf.stack(Q_value_perAct))
Target_Q_value=tf.stop_gradient(Target_Q_value)
Loss=tf.losses.huber_loss(labels=(Rew+gamma*tf.reduce_max(Target_Q_value,axis=1)), predictions=Q_value_perAct)
Loss=tf.reduce_mean(Loss)
global_step=tf.placeholder(tf.float32, shape=())
learning_rate_use = tf.train.exponential_decay(lr_eve, global_step, 10000, 0.98, staircase=True)
trainer=tf.train.RMSPropOptimizer(learning_rate_use,0.99,0.0,1e-6)
gvs = trainer.compute_gradients(Loss)

#Clip gradient to avoid overlapping
def ClipIfNotNone(grad):
    if grad is None:
        return grad
    return tf.clip_by_value(grad, -1.0, 1.0)
clipped_gradients = [(ClipIfNotNone(grad), var) for grad, var in gvs]
train_step = trainer.apply_gradients(clipped_gradients)
Process=StateProcessor()


#handle reward
def Deal_reward(gamma,Reward):
		#Give all 0 reward the last one (+1 or -1)
	count=len(Reward)
	start=0
	while count>0:
		count=count-1
		if Reward[count]!=0 :
			start=Reward[count]
			Base_count=count
			continue
		if start!=0 and Reward[count]==0:
			#Reward[count]=gamma**(Base_count-count)*start
			Reward[count]=start
			
			continue
	return Reward



# Gathering the initial training dataset
with tf.Session() as sess:
	if N_Q==0:
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()
		saver.save(sess,'./save/%s%d.ckpt' % (model_method, N_Q))
	step=0
	stop=0
	raw_record={}
	Observation=[]
	Reward=[]
	Action=[]
	while step<(initial_num_in_episode*(N_frame+N_delay-1)) or stop==0:
		Initial=env.reset()
		start_key=step
		n_delay=0
		done=False
		action=env.action_space.sample()
		while not done:
			if (len(Observation)==N_frame):
				n_delay=0
				
			step=step+1
			n_delay=n_delay+1
			I=Process.process(sess,Initial)
			Observation.append(np.reshape(I.astype(np.float).ravel(),(1,7056)))
			if n_delay>N_delay:
				action=env.action_space.sample()
				n_delay=0
			observation, reward, done, info = env.step(action)
			if not (len(Observation)<N_frame):
				Reward.append(reward)
				Action.append(action)
			Initial=observation
		if step>(initial_num_in_episode*(N_frame+N_delay-1)):
			stop=1
	Reward=Deal_reward(gamma,Reward)
	Observation=np.squeeze(np.array(Observation))
	for j in range(initial_num_in_episode):
		i=j+1
		m=(N_frame+N_delay-1)*j
		raw_record[str(i)]=(Observation[m:(m+N_frame)],sum(Reward[(m):(m+N_delay)]),Action[m],Observation[(m+(N_delay)):(m+(N_delay)+N_frame)])


# learn by play
while N_Q<Max_N_Q:
	lr_eve=lr*((0.98)**(N_Q))
	if lr_eve<0.000000025:
		lr_eve=0.000000025
	if N_Q<100:
		sigma=(1000000-N_Q*10000)/1000000.0*0.9+0.1
	else:
		sigma=0.1
	print (sigma)
	print (lr_eve)
	aver_score=[]
	# Define the graph again, since one can comment initial gathering when the model has been trained
	if N_Q>0:
		tf.reset_default_graph()
		with tf.name_scope('main_network'):
			Q=Q_function(batch_size=batch_size,n_frame=N_frame,action_num=6)
			Q_value=Q.CNN_part()

		with tf.name_scope('target_network'):
			Target_Q=Q_function(batch_size=batch_size,n_frame=N_frame,action_num=6)
			Target_Q_value=Target_Q.CNN_part()
		main_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES , scope='main_network') 
		target_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES , scope='target_network')
		assign_ops = []
		for main_var, target_var in zip(main_variables, target_variables):
			assign_ops.append(tf.assign(target_var, tf.identity(main_var)))
		copy_operation = tf.group(*assign_ops)
		Act=tf.placeholder(tf.int32,shape=(batch_size,))
		Rew=tf.placeholder(tf.float32,shape=(batch_size,))
		Q_value_perAct=[]
		for i in range(batch_size):
			Q_value_perAct.append(Q_value[i,Act[i]])
		Q_value_perAct=tf.transpose(tf.stack(Q_value_perAct))
		Target_Q_value=tf.stop_gradient(Target_Q_value)
		Loss=tf.losses.huber_loss(labels=(Rew+gamma*tf.reduce_max(Target_Q_value,axis=1)), predictions=Q_value_perAct)
		Loss=tf.reduce_mean(Loss)
		global_step=tf.placeholder(tf.float32, shape=())
		learning_rate_use = tf.train.exponential_decay(lr_eve, global_step,10000, 0.98, staircase=True)
		trainer=tf.train.RMSPropOptimizer(learning_rate_use,0.99,0.0,1e-6)
		gvs = trainer.compute_gradients(Loss)
		def ClipIfNotNone(grad):
		    if grad is None:
		        return grad
		    return tf.clip_by_value(grad, -1, 1)
		clipped_gradients = [(ClipIfNotNone(grad), var) for grad, var in gvs]
		train_step = trainer.apply_gradients(clipped_gradients)
		Process=StateProcessor()
	

	# Start to learn by play
	with tf.Session() as sess:
		checkpoint_path = './save/%s%d.ckpt' % (model_method, N_Q)
		#saver = tf.train.import_meta_graph('./save/%s%d.ckpt' % (model_method,0)+'.meta')
		saver = tf.train.Saver()
		saver.restore(sess, checkpoint_path)
		tf.get_default_graph().finalize()
		sess.run(copy_operation)
		print (N_Q)
		count=0
		Initial=env.reset()
		I=Process.process(sess,Initial)
		Play_input_data=[np.reshape(I.astype(np.float).ravel(),(1,7056))]
		State_now=[]
		Reward=[]
		Mark_reward=[]
		Action=[]
		State_next=[]
		Accumulate_reward=[]
		while count<max_train_step:
			count=count+1
			while len(Play_input_data)<N_frame:
				action=env.action_space.sample()
				env.render()
				observation, reward, done, info = env.step(action)
				if abs(reward)==1:
					Accumulate_reward.append(reward)
				Initial=observation
				I=Process.process(sess,Initial)
				Play_input_data.append(np.reshape(I.astype(np.float).ravel(),(1,7056)))
				Reward.append(reward)
				if done==True:
					aver_score.append(sum(Accumulate_reward))
					print (sum(Accumulate_reward))
					Accumulate_reward=[]
					env.reset()
			assert len(Play_input_data)==N_frame
			Play_input_data_ar=np.squeeze(np.array(Play_input_data))
			State_now.append(Play_input_data_ar)
			Play_input_data_ar=np.reshape(Play_input_data_ar,(1,N_frame,7056))
			lin=random.uniform(0,1)
			if lin<0:
				action=env.action_space.sample()
			else:
				Real_Q_value=sess.run(Q_value,feed_dict={Q.Input_tensor:Play_input_data_ar})
				action=np.argmax(Real_Q_value)
			Action.append(int(action))
			for i in range(N_delay):
				env.render()
				observation, reward, done, info = env.step(action)
				if abs(reward)==1:
					Accumulate_reward.append(reward)
				Reward.append(reward)
				del Play_input_data[0]
				Initial=observation
				I=Process.process(sess,Initial)
				Play_input_data.append(np.reshape(I.astype(np.float).ravel(),(1,7056)))
				if done==True:
					aver_score.append(sum(Accumulate_reward))
					print (sum(Accumulate_reward))
					Accumulate_reward=[]
					env.reset()
			Mark_reward.append(len(Reward)-1)
			State_next.append(np.squeeze(np.array(Play_input_data)))
			Train_data=[]
			Reward_train=[]
			Action_train = []
			Target_Train_data=[]
			for i in range(batch_size):
				eps=random.randint(1,initial_num_in_episode)
				Train_data.append(raw_record[str(eps)][0])
				Reward_train.append(raw_record[str(eps)][1])
				Action_train.append(raw_record[str(eps)][2])
				Target_Train_data.append(raw_record[str(eps)][3])
			Train_data=np.array(Train_data)
			Reward_train=np.array(Reward_train)
			Action_train=np.array(Action_train)
			Target_Train_data=np.array(Target_Train_data)        
			train_step.run(feed_dict={ global_step: count, Q.Input_tensor:Train_data,Target_Q.Input_tensor: Target_Train_data, Rew: Reward_train,Act:Action_train})
		
			#print (Loss_batch.eval(feed_dict={Q.Input_tensor:Train_data,Target_Q.Input_tensor: Target_Train_data, Rew: Reward_train,Act:Action_train}))
		Reward=Deal_reward(gamma,Reward)
		Reward_mem=[]
		for i in Mark_reward:
			Reward_mem.append(sum(Reward[(i+1-N_delay):(1+i)]))
		assert len(Reward_mem)==len(Action)
		assert len(Action)==len(State_now)
		assert len(State_now)==len(State_next)

		for i in range(initial_num_in_episode):
			if (i+len(State_now))<initial_num_in_episode:
				raw_record[str(i)]=raw_record[str(i+len(State_now))]
			else:
				j=len(State_now)+i-initial_num_in_episode
				raw_record[str(i)]=(State_now[j],Reward_mem[j],Action[j],State_next[j])
		
		checkpoint_path = './save_10/%s%d.ckpt' % (model_method, (N_Q+1))
		saver.save(sess, checkpoint_path, write_meta_graph=False)
		N_Q=N_Q+1
		print (np.mean(aver_score))
