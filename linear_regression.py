import tensorflow as tf;

# 1. Build graph using TF operations (H(x) = Wx + b)
x_train = [1,2,3];
y_train = [1,2,3];

# variable 은 trainable value. (tensorflow 가 사용하는 변수)
W = tf.Variable(tf.random_normal([1]), name='weight');
b = tf.Variable(tf.random_normal([1]), name='bias');
# Our hypothesis Wx+b
hypothesis = x_train * W + b;

#cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_train));

#Minimize(GradientDescent)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01);
train = optimizer.minimize(cost);

#2.3. Run/update grapth and get results
#Launch the graph in a session
sess = tf.Session();
# 위에서  W, b 라는 variable 을 사용했다. 이를 사용하기 위해선 반드시 아래의
# global_variables_initializer() 함수를 사용해야 한다.
sess.run(tf.global_variables_initializer());

#Fit the line
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))

