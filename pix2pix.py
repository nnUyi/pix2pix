import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
import time
from ops import *
from glob import glob
from utils import *

class pix2pix():
    model_name = 'pix2pix'
    
    def __init__(self, config, batch_size=1, input_height=256, input_width=256, input_channels=3, df_dim=64, gf_dim=64, sess=None):
        self.batch_size = batch_size
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        
        self.config = config
        
        self.sess = sess

    def generator_unet(self, input_x, scope_name='generator', reuse=False):
        with tf.variable_scope(scope_name) as scope:
            if reuse:
                scope.reuse_variables()
            
            with slim.arg_scope([slim.conv2d_transpose, slim.conv2d],
                                #weights_regularizer = slim.l2_regularizer(0.05),
                                weights_initializer = tf.truncated_normal_initializer(stddev=0.02),
                                activation_fn = None,
                                normalizer_fn = slim.batch_norm,
                                padding='SAME'):
                
                conv1 = leaky_relu(slim.conv2d(input_x, self.gf_dim, [5,5], stride=2, normalizer_fn=None, scope='g_conv1'))
                conv2 = leaky_relu(slim.conv2d(conv1, self.gf_dim*2, [5,5], stride=2, scope='g_conv2'))
                conv3 = leaky_relu(slim.conv2d(conv2, self.gf_dim*4, [5,5], stride=2, scope='g_conv3'))
                conv4 = leaky_relu(slim.conv2d(conv3, self.gf_dim*8, [5,5], stride=2, scope='g_conv4'))
                conv5 = leaky_relu(slim.conv2d(conv4, self.gf_dim*8, [5,5], stride=2, scope='g_conv5'))
                conv6 = leaky_relu(slim.conv2d(conv5, self.gf_dim*8, [5,5], stride=2, scope='g_conv6'))
                conv7 = leaky_relu(slim.conv2d(conv6, self.gf_dim*8, [5,5], stride=2, scope='g_conv7'))
                
                conv8 = slim.conv2d(conv7, self.gf_dim*8, [5,5], stride=2, activation_fn=None, scope='g_conv8')
                
                dconv1 = slim.conv2d_transpose(tf.nn.relu(conv8), self.gf_dim*8, [5,5], stride=2, activation_fn=None, scope='g_dconv1')
                dconv1 = tf.nn.dropout(dconv1, 0.5)
                dconv1 = tf.concat([dconv1, conv7], 3)
                
                dconv2 = slim.conv2d_transpose(tf.nn.relu(dconv1), self.gf_dim*8, [5,5], stride=2, activation_fn=None, scope='g_dconv2')
                dconv2 = tf.nn.dropout(dconv2, 0.5)
                dconv2 = tf.concat([dconv2, conv6], 3)
                
                dconv3 = slim.conv2d_transpose(tf.nn.relu(dconv2), self.gf_dim*8, [5,5], stride=2, activation_fn=None, scope='g_dconv3')
                dconv3 = tf.nn.dropout(dconv3, 0.5)
                dconv3 = tf.concat([dconv3, conv5], 3)
                
                dconv4 = slim.conv2d_transpose(tf.nn.relu(dconv3), self.gf_dim*8, [5,5], stride=2, activation_fn=None, scope='g_dconv4')
                #dconv4 = tf.nn.dropout(dconv4, 0.5)
                dconv4 = tf.concat([dconv4, conv4], 3)
                
                dconv5 = slim.conv2d_transpose(tf.nn.relu(dconv4), self.gf_dim*4, [5,5], stride=2, activation_fn=None, scope='g_dconv5')
                #dconv5 = tf.nn.dropout(dconv5, 0.5)
                dconv5 = tf.concat([dconv5, conv3], 3)
                
                dconv6 = slim.conv2d_transpose(tf.nn.relu(dconv5), self.gf_dim*2, [5,5], stride=2, activation_fn=None, scope='g_dconv6')
                #dconv6 = tf.nn.dropout(dconv6, 0.5)
                dconv6 = tf.concat([dconv6, conv2], 3)
                # 128
                dconv7 = slim.conv2d_transpose(tf.nn.relu(dconv6), self.gf_dim, [5,5], stride=2, activation_fn=None, scope='g_dconv7')
                #dconv7 = tf.nn.dropout(dconv7, 0.5)
                dconv7 = tf.concat([dconv7, conv1], 3)
                # 256
                out = slim.conv2d_transpose(tf.nn.relu(dconv7), self.input_channels, [5,5], stride=2, normalizer_fn=None, activation_fn=tf.nn.tanh, scope='g_out')
                print(out)
                return out
                
    def discriminator(self, input_x, scope_name='discriminator', reuse=False):
        with tf.variable_scope(scope_name) as scope:
            if reuse:
                scope.reuse_variables()
            with slim.arg_scope([slim.conv2d],
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                #weights_regularizer=slim.l2_regularizer(0.05),
                                activation_fn = None,
                                normalizer_fn = slim.batch_norm,
                                padding='SAME'):
                # 256->128
                conv1 = leaky_relu(slim.conv2d(input_x, self.df_dim, [5,5], stride=2, normalizer_fn=None, scope='d_conv1'))
                print(conv1)
                # 128->64
                conv2 = leaky_relu(slim.conv2d(conv1, self.df_dim*2, [5,5], stride=2, scope='d_conv2'))
                print(conv2)
                # 64->32
                conv3 = leaky_relu(slim.conv2d(conv2, self.df_dim*4, [5,5], stride=2, scope='d_conv3'))
                print(conv3)
                # 32->31
                #conv3 = tf.pad(conv3, [[0,0],[1,1],[1,1],[0,0]], mode='CONSTANT')
                conv4 = leaky_relu(slim.conv2d(conv3, self.df_dim*8, [5,5], stride=1, scope='d_conv4'))
                print(conv4)
                # 31->30
                #conv4 = tf.pad(conv4, [[0,0],[1,1],[1,1],[0,0]], mode='CONSTANT')
                #conv5 = slim.conv2d(conv4, 1, [4,4], stride=1, normalizer_fn=None, activation_fn=None, padding='VALID', scope='d_conv5')
                conv4_flat = tf.reshape(conv4, [self.batch_size, -1])
                fc1 = slim.fully_connected(conv4_flat, 1, normalizer_fn=None, activation_fn=None, scope='d_fc1')
                print(fc1)
                return(fc1)
                #return conv5
    
    def build_model(self):
        self.input_A = tf.placeholder(tf.float32, [self.batch_size, self.input_height, self.input_width, self.input_channels], name='input_A')
        self.input_B = tf.placeholder(tf.float32, [self.batch_size, self.input_height, self.input_width, self.input_channels], name='input_B')
        
        self.input_AB = tf.concat([self.input_A, self.input_B], 3)
        assert self.input_AB.get_shape().as_list() == [self.batch_size, self.input_height, self.input_width, self.input_channels*2]
        
        self.D_real_logits = self.discriminator(self.input_AB, reuse=False)
        self.fake_B = self.generator_unet(self.input_A, reuse=False)
        
        self.fake_AB = tf.concat([self.input_A, self.fake_B], 3)
        
        self.D_fake_logits = self.discriminator(self.fake_AB, reuse=True)
        
        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)
                
        self.D_real_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_real_logits, tf.ones_like(self.D_real_logits)))
        self.D_fake_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_fake_logits, tf.zeros_like(self.D_fake_logits)))        
        self.d_loss = self.D_real_loss + self.D_fake_loss

        self.l1_loss = tf.reduce_mean(tf.abs(self.fake_B-self.input_B))
        self.G_adv_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_fake_logits, tf.ones_like(self.D_fake_logits)))
        self.g_loss = self.config.lambd*self.l1_loss + self.G_adv_loss
        
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]
        
        self.d_optimization = tf.train.AdamOptimizer(learning_rate=self.config.lr, beta1=self.config.beta1, beta2=self.config.beta2).minimize(self.d_loss, var_list=d_vars)
        
        self.g_optimization = tf.train.AdamOptimizer(learning_rate=self.config.lr, beta1=self.config.beta1, beta2=self.config.beta2).minimize(self.g_loss, var_list=g_vars)
        
        self.l1_loss_summary = tf.summary.scalar('l1_loss', self.l1_loss)
        self.d_loss_summary = tf.summary.scalar('d_loss', self.d_loss)
        self.g_loss_summary = tf.summary.scalar('g_loss', self.g_loss)
        self.summaries = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter('logs', self.sess.graph)
        
        # save model
        self.saver = tf.train.Saver()
                
    def train(self):
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()
        data_list = glob(os.path.join(self.config.dataset_dir, self.config.dataset_name, self.config.phase, '*.*'))
        batch_idxs = int(len(data_list)/self.batch_size)
        
        counter = 0
        check_bool, counter = self.load_model(self.config.checkpoint_dir)
        if check_bool:
            print('[!!!] load model successfully')
            counter = counter+1
        else:
            print('[***] fail to load model')
            counter = 1
        
        start_time = time.time()
        for epoch in range(self.config.epoches):
            for idx in range(batch_idxs):
                batch_files = data_list[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_x = [get_image(batch_file) for batch_file in batch_files]
                batch_x = np.array(batch_x).astype(np.float32)
                input_B = batch_x[:,:,:self.input_width,:]
                input_A = batch_x[:,:,self.input_width:,:]

                _, d_loss, summaries = self.sess.run([self.d_optimization, self.d_loss, self.summaries], feed_dict={self.input_A:input_A,
                                                                                             self.input_B:input_B})
                _, g_loss, l1_loss, summaries = self.sess.run([self.g_optimization, self.g_loss, self.l1_loss, self.summaries], feed_dict={self.input_A:input_A,self.input_B:input_B})
                #_, g_loss, l1_loss, summaries = self.sess.run([self.g_optimization, self.g_loss, self.l1_loss, self.summaries], feed_dict={self.input_A:input_A,self.input_B:input_B})
                counter=counter+1
                end_time = time.time()
                total_time = end_time - start_time
                print('epoch{}[{}/{}]:phase:{}, total_time:{:.4f}, d_loss:{:.4f}, g_loss:{:.4f}, l1_loss:{:.4f}'.format(epoch, idx, batch_idxs, self.config.phase, total_time, d_loss, g_loss, self.config.lambd*l1_loss))
                
                self.summary_writer.add_summary(summaries, global_step=counter)

                if np.mod(counter, 100)==0:
                    self.sample(self.config.sample_dir, epoch, idx)
                if np.mod(counter, 500)==0:
                    self.save_model(self.config.checkpoint_dir, counter)
    
    def sample(self, sample_dir, epoch, idx):
        input_A, input_B = self.load_sample()
        sample_B = self.sess.run(self.fake_B, feed_dict={self.input_A:input_A, self.input_B:input_B})
        sample = np.concatenate([input_A, input_B, sample_B], 2)
        save_images(sample, [1,1], '{}/{}_{}_{:04d}_{:04d}.png'.format(self.config.sample_dir,self.config.dataset_name, self.config.phase, epoch, idx))
    
    def load_sample(self):
        batch_files = np.random.choice(glob(os.path.join(self.config.dataset_dir, self.config.dataset_name, 'val', '*.*')), self.batch_size)
        batch_data = [get_image(batch_file) for batch_file in batch_files]
        batch_data = np.array(batch_data).astype(np.float32)
        input_A = batch_data[:,:,self.input_width:,:]
        input_B = batch_data[:,:,:self.input_width,:]
        return input_A, input_B

    def test(self):
        data_list = glob(os.path.join(self.config.dataset_dir, self.config.dataset_name, self.config.phase, '*.*'))
        batch_idxs = int(len(data_list)/self.batch_size)
        print('test') 
        counter = 0
        check_bool, counter = self.load_model(self.config.checkpoint_dir)
        if check_bool:
            print('[!!!] load model successfully')
        else:
            print('[***] fail to load model')
            return

        for idx in range(batch_idxs):
            batch_files = data_list[idx*self.batch_size:(idx+1)*self.batch_size]
            batch_x = [get_image(batch_file) for batch_file in batch_files]
            batch_x = np.array(batch_x).astype(np.float32)
            input_B = batch_x[:,:,:self.input_width,:]
            input_A = batch_x[:,:,self.input_width:,:]
            #input_B = np.random.normal(-1,1,[1,256,256,3])
            #print(batch_files)
            sample_B = self.sess.run(self.fake_B, feed_dict={self.input_A:input_A})
            sample = np.concatenate([input_A, input_B, sample_B], 2)
            save_images(sample, [1,1], '{}/{}_{}_{:04d}.png'.format(self.config.test_dir, self.config.dataset_name, self.config.phase, idx))
            #save_images(batch_x, [1,1], '{}/{}_{}_{:04d}.png'.format(self.config.test_dir, self.config.dataset_name, 'real', idx))
            print('testing:{}'.format(idx))
            
    def valuate(self, sample_dir, epoch, idx):        
        pass
        
    # save model            
    @property
    def model_dir(self):
        return "{}_{}_{}".format(
            self.model_name, self.config.dataset_name,
            self.batch_size)

    def save_model(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load_model(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

if __name__=='__main__':
    input_x = np.random.normal(-1,1, [64,256,256,3]).astype(np.float32)
    gan = pix2pix(None)
    gan.discriminator(input_x)
    gan.generator_unet(input_x)
