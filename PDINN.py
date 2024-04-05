''' This code combines the principle of Peridynamics with the concept of Physics-Informed Neural Networks (PINNs) to
    predict the displacement field of a cracked plate.
    The Ref. article for the PD-INN is (Eghbalpoor R. and Sheidaei A., 2024).
    The following code simulates a plate subjected to a tensile load at the right edge. A pre-crack is considered within the domain.
    The plate is discretized into a set of material points. The displacement field is predicted using a neural network and the loss function
    is defined based on the principle of Peridynamics.
    The code is tested on a single NVIDIA GeForce RTX 3060 Laptop GPU with Core(TM) i7-12650H. The code runtime was ~10min (dxdy=0.002).
    The code is tested on a single NVidia A100 in the High Performance Computing facility at Iowa State University. The code runtime was ~9min (dxdy=0.001).
    Reaserchers are encourages to develop the code and apply changes to simulate crack propagation by considering transfer learning technique and
    partial training strategy as metioned in the paper.
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import cg

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
DTYPE='float64'
tf.keras.backend.set_floatx(DTYPE)

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only allocate 10GB of memory on the first GPU
#   try:
#     tf.config.set_logical_device_configuration(
#         gpus[0],
#         [tf.config.LogicalDeviceConfiguration(memory_limit=10000)])
#     logical_gpus = tf.config.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Virtual devices must be set before GPUs have been initialized
#     print(e)

def init_model(num_points):
    num_hidden_layers=6
    num_neurons_per_layer=64
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer((2*num_points,)))
    for i in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(num_neurons_per_layer,
            activation=tf.keras.activations.get('tanh'),
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.001)))
    model.add(tf.keras.layers.Dense(2*num_points))
    return model

def peridynamic_bond_c(inputs,thickness, horizon):
    E_young1=192e9
    G0=83e3
    bond_c0=9*E_young1/3.1415/thickness/horizon**3
    critical_stretch=tf.sqrt(4.*3.1415*G0/9./E_young1/horizon)
    return bond_c0, tf.cast(critical_stretch , tf.float64)

def peridynamic_kernel(inputs, horizon):
    x, y = inputs[:, 0:1], inputs[:, 1:2]
    Xs=tf.tile(x, [1,tf.shape(inputs)[0]])
    Ys=tf.tile(y, [1,tf.shape(inputs)[0]])
    Distance=tf.sqrt(tf.square(Xs-tf.transpose(Xs))+tf.square(Ys-tf.transpose(Ys)))
    eff_len = tf.where(tf.less(Distance, horizon), tf.exp(-4*Distance**2/horizon**2), 0)
    bond_dirX = (tf.transpose(Xs)-Xs)/Distance
    bond_dirY = (tf.transpose(Ys)-Ys)/Distance
    bond_dirX = tf.where(tf.math.is_nan(bond_dirX),0,bond_dirX)
    bond_dirY = tf.where(tf.math.is_nan(bond_dirY),0,bond_dirY)
    return Distance, eff_len, bond_dirX, bond_dirY

def precrack(inputs, precrack_pos, eff_len):
    x, y = inputs[:, 0:1], inputs[:, 1:2]
    Xs = tf.transpose(tf.tile(x, [1,tf.shape(inputs)[0]]))
    Ys = tf.transpose(tf.tile(y, [1,tf.shape(inputs)[0]]))
    crack_start = precrack_pos[0,:]
    crack_end = precrack_pos[1,:]
    m2 = (crack_end[1] - crack_start[1]) / (crack_end[0] - crack_start[0])
    c2 = crack_start[1] - m2 * crack_start[0]
    m1 = (Ys-y)/(Xs-x)
    m1 = tf.where(tf.math.is_inf(m1), 10000., m1)
    c1 = y - m1 * x
    x_intersection = (c2 - c1) / (m1 - m2)
    y_intersection = m1 * x_intersection + c1
    dist1 = tf.sqrt(tf.square(x_intersection-crack_start[0]) + tf.square(y_intersection-crack_start[1]))
    dist2 = tf.sqrt(tf.square(x_intersection-crack_end[0]) + tf.square(y_intersection-crack_end[1]))
    dist3 = tf.sqrt(tf.square(crack_start[0]-crack_end[0]) + tf.square(crack_start[1]-crack_end[1]))
    damage1 = tf.where(tf.less(tf.abs(dist1+dist2-dist3),1.e-6),1.,0.)
    dist1 = tf.sqrt(tf.square(x_intersection-x) + tf.square(y_intersection-y))
    dist2 = tf.sqrt(tf.square(x_intersection-Xs) + tf.square(y_intersection-Ys))
    dist3 = tf.sqrt(tf.square(Xs-x) + tf.square(Ys-y))
    damage2 = tf.where(tf.less(tf.abs(dist1+dist2-dist3),1.e-6),1.,0.)
    damage = tf.where(tf.equal(damage1,damage2),damage1,0.)
    damage = tf.where(tf.equal(eff_len,0.),0.,damage)
    return tf.cast(damage , tf.float64)

def compute_neighbors(i):
        v1 = np.zeros(29,'int16')-1
        k = 0
        for j in range(npointsnp):
            distance = np.sqrt((inputsnp[i, 0] - inputsnp[j, 0]) ** 2 + (inputsnp[i, 1] - inputsnp[j, 1]) ** 2)
            if 0 < distance < horizon:
                v1[k] = j
                k += 1
        return v1

def compute_stiffness(k):
    SS1=np.zeros(2*npointsnp)
    i=k//2
    B_D=damagenp[i,:]
    for j in range(29):
        if neighbornp[i,j] != -1:
            vector=-1*np.array([[inputsnp[i,0]-inputsnp[neighbornp[i,j],0]],[inputsnp[i,1]-inputsnp[neighbornp[i,j],1]]])
            BondLengthScalar=np.linalg.norm(vector)
            B_L_Eff = np.exp(-4*BondLengthScalar**2/horizon**2)
            const = bond_c*volume**2/BondLengthScalar**3*B_L_Eff*(1-B_D[neighbornp[i,j]])
            a1=const*vector[0]*vector[0]
            a2=const*vector[0]*vector[1]
            b1=const*vector[1]*vector[0]
            b2=const*vector[1]*vector[1]
            if k % 2 == 0:
                SS1[2*i]=SS1[2*i]+a1
                SS1[2*i+1]=SS1[2*i+1]+a2
                SS1[2*neighbornp[i,j]]=-a1
                SS1[2*neighbornp[i,j]+1]=-a2
            else:
                SS1[2*i]=SS1[2*i]+b1
                SS1[2*i+1]=SS1[2*i+1]+b2
                SS1[2*neighbornp[i,j]]=-b1
                SS1[2*neighbornp[i,j]+1]=-b2
    return SS1

def update_RHS_Stiff(npointsnp, dxdy, inputsnp, u_BC, x_max, x_min, Stiffnessnp):
    RHS1np = np.zeros((2*npointsnp,1))
    deltaX = u_BC
    for k in range(2*npointsnp):
        i=k//2
        if inputsnp[i,0] > x_max-2.2*dxdy:
            if k % 2 ==0:
                RHS1np += Stiffnessnp[:,k:k+1]*deltaX
    for k in range(2*npointsnp):
        i=k//2
        if inputsnp[i,0] > x_max-2.2*dxdy:
            if k % 2 ==0:
                RHS1np[k,0] = deltaX
            else:
                RHS1np[k,0] = 0
    for k in range(2*npointsnp):
        i=k//2
        SS1=np.zeros(2*npointsnp)
        if inputsnp[i,0] > x_max-2.2*dxdy or inputsnp[i,0] < x_min+2.2*dxdy:
            Stiffnessnp[k:k+1,:]=SS1
    Stiffnessnp = np.transpose(Stiffnessnp)
    for k in range(2*npointsnp):
        i=k//2
        SS1=np.zeros(2*npointsnp)
        SS1[k]=-1
        if inputsnp[i,0] > x_max-2.2*dxdy or inputsnp[i,0] < x_min+2.2*dxdy:
            Stiffnessnp[k:k+1,:]=SS1
    return  RHS1np, Stiffnessnp

def pinn_loss(output, inputs, id_innerMPs, id_BC0, id_BC1, id_damage, bond_c, Distance, eff_len, bond_dirX, bond_dirY, volume, damage, epoch, u_BC, uxy_linearized):

    Distance_denom = tf.where(tf.equal(Distance,0), tf.constant(1., dtype=tf.float64), Distance)
    new_coords = output + inputs[:, 0:2]
    x_new, y_new = new_coords[:, 0:1], new_coords[:, 1:2]
    Xs=tf.tile(x_new, [1,tf.shape(inputs)[0]])
    Ys=tf.tile(y_new, [1,tf.shape(inputs)[0]])
    Distance_new = tf.sqrt(tf.square(Xs-tf.transpose(Xs))+tf.square(Ys-tf.transpose(Ys)))
    stretch = (Distance_new-Distance)/Distance_denom
    Distance_new_denom = tf.where(tf.equal(Distance_new,0), tf.constant(1., dtype=tf.float64), Distance_new)
    bond_dirX = (tf.transpose(Xs)-Xs)/Distance_new_denom
    bond_dirY = (tf.transpose(Ys)-Ys)/Distance_new_denom
    forceX = bond_c * volume**2 * tf.reduce_sum(stretch * bond_dirX * (eff_len) * (1.-(damage)), axis=1, keepdims=True)
    forceY = bond_c * volume**2 * tf.reduce_sum(stretch * bond_dirY * (eff_len) * (1.-(damage)), axis=1, keepdims=True)
    force=tf.sqrt(tf.reduce_sum(tf.square(tf.reshape(tf.stack([forceX,forceY], axis=1),[tf.shape(forceX)[0],2])),axis=1))
    force=tf.reshape(force,[-1,1])
    force_inner = tf.gather(force,id_innerMPs)
    force_dmg = tf.gather(force,id_damage)
    force_grad = tf.where(tf.math.logical_and(output[:,0:1] > 0.35*u_BC, output[:,0:1] < 0.65*u_BC),force,0.)

    """ these commands could be used to calculate force_grad.
    You need to convert output vector to output_image.
    [Iy, Ix] = np.gradient(output_image[:,:].numpy())
    Ix = tf.convert_to_tensor(Ix)
    Ix = tf.reshape(tf.reverse(Ix[:,:],[0]),(-1,1))
    Ix = tf.cast(Ix, tf.float64)
    force_grad = tf.where(Ix > 0.7* tf.reduce_max(Ix),force,0.)
    """
    u_pred_BC0 = tf.gather(output,id_BC0)
    u_pred_BC1 = tf.gather(output,id_BC1)

    """ You might want to adjust the coefficients  
    below for different material properties and domains
    """
    if epoch < 50000:
        c1 = 1.e-3
        c2 = 500000.
    elif epoch < 100000:
        c1 = 1.e-2
        c2 = 2. * 500000.
    elif epoch < 200000:
        c1 = 0.1
        c2 = 4. * 500000.
    else:
        c1 = 4.
        c2 = 10. * 500000.
    c5 = 1.5/(1.+tf.exp(-(epoch-(epoch//5000)*5000-2500)/700))
    c5 = c1 * tf.cast(c5, tf.float64)
    loss1 = c1 * tf.norm(force_inner)
    loss2 = c2 * ((tf.norm(u_pred_BC0[:, 0:1])) + (tf.norm(u_pred_BC0[:, 1:2])) + \
                    (tf.norm(u_pred_BC1[:, 0:1]-u_BC)) + (tf.norm(u_pred_BC1[:, 1:2])))
    loss3 = c1 * tf.norm(force_dmg)
    # loss4 = tf.norm(tf.matmul(Stiffness, output)+RHS1)
    loss4 = c2 * (tf.norm(output[:, 0:1]-uxy_linearized[:, 0:1]) + tf.norm(output[:, 1:2]-uxy_linearized[:, 1:2]))
    loss5 = c5 * tf.norm(force_grad)

    loss = loss1 + loss2 + loss3 + loss4 + loss5
    
    if epoch % 1000 == 0:
        tf.print(f"epoch {epoch}, Loss= {loss.numpy():.6f}, Loss1= {loss1.numpy():.6f}, Loss2= {loss2.numpy():.6f}, Loss3= {loss3.numpy():.6f}, Loss4= {loss4.numpy():.6f}, Loss5= {loss5.numpy():.6f}")

    return loss

def train_step(inputs, inputs2, model, optimizer, id_innerMPs, id_BC0, id_BC1, id_damage, bond_c, Distance, eff_len, bond_dirX, bond_dirY, volume, damage, epoch, u_BC, uxy_all_data):
    with tf.GradientTape() as tape:
        output = tf.transpose(model(tf.transpose(inputs2)))
        output = tf.reshape(output , [tf.shape(inputs)[0],2])
        loss = pinn_loss(output, inputs, id_innerMPs, id_BC0, id_BC1, id_damage, bond_c, Distance, eff_len, bond_dirX, bond_dirY, volume, damage, epoch, u_BC, uxy_all_data)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # del tape
    return loss

#======================================================================================#

dxdy=0.002
horizon=3.015*dxdy
thickness=0.001
volume=dxdy**2*thickness
x_max = 0.1
x_min = 0.
y_max = 0.05
u_BC_0 = 0.0002


x = tf.range(0.,x_max+dxdy/2,dxdy,'float64')
y = tf.range(0.,y_max+dxdy/2,dxdy,'float64')
X, Y = tf.meshgrid(x, y)
inputs = tf.stack([tf.reshape(X, [-1]), tf.reshape(Y, [-1])], axis=1)
npoints = tf.shape(inputs)[0]
inputs2 = tf.reshape(inputs, [-1])
inputs2 = tf.reshape(inputs2, (-1,1))

Distance, eff_len, bond_dirX, bond_dirY = peridynamic_kernel(inputs, horizon)
bond_c, critical_stretch = peridynamic_bond_c(inputs,thickness, horizon)

crack_pos_x=0.02
crack_len=0.01
precrack_x = -1000.
precrack_y = -1000.
for i in range (npoints):
    if tf.abs(inputs[i,0]-crack_pos_x)<dxdy*0.499:
        precrack_x = inputs[i,0] + dxdy/2.
        break
    else:
        precrack_x = tf.convert_to_tensor(crack_pos_x)
for i in range (npoints):
    if tf.abs(inputs[i,1]-crack_len)<dxdy*0.499:
        precrack_y = inputs[i,1] + dxdy/1.99
        break
    else:
        precrack_y = tf.convert_to_tensor(crack_len)
precrack_all = tf.constant([[precrack_x.numpy(),tf.constant(0., dtype=tf.float64).numpy()],[(precrack_x+0.0001*precrack_x).numpy(),precrack_y.numpy()]], dtype=tf.float64)
damage = precrack(inputs, precrack_all, eff_len)
netDamage = tf.reduce_sum(damage, axis=1) / (tf.reduce_sum(tf.where(tf.equal((eff_len),0.),tf.constant(0., dtype=tf.float64),tf.constant(1., dtype=tf.float64)), axis=1)-1.)
id_damage = tf.where(netDamage > 0)
id_damage = tf.reshape(id_damage[:,0:1],tf.shape(id_damage)[0])


id_innerMPs = tf.where(tf.math.logical_and(inputs[:,0:1] > tf.reduce_min(x)+2.015*dxdy, inputs[:,0:1] < (tf.reduce_max(x)-2.015*dxdy)))
id_innerMPs = tf.reshape(id_innerMPs[:,0:1],tf.shape(id_innerMPs)[0])
id_BC0 = tf.where(inputs[:,0:1] < tf.reduce_min(x)+2.015*dxdy)
id_BC0 = tf.reshape(id_BC0[:,0:1],tf.shape(id_BC0)[0])
id_BC1 = tf.where(inputs[:,0:1] > (tf.reduce_max(x)-2.015*dxdy))
id_BC1 = tf.reshape(id_BC1[:,0:1],tf.shape(id_BC1)[0])
id_non_Damage = tf.where(tf.math.logical_and(inputs[:,0:1] > tf.reduce_min(x)+2.015*dxdy, inputs[:,0:1] < (tf.reduce_max(x)-2.015*dxdy)),tf.constant(1., dtype=tf.float64),tf.constant(0., dtype=tf.float64))


xnp = np.arange(0.,x_max+dxdy/2,dxdy,'float64')
ynp = np.arange(0.,y_max+dxdy/2,dxdy,'float64')
Xnp, Ynp = np.meshgrid(xnp, ynp)
inputsnp = np.stack([np.reshape(Xnp, [-1]), np.reshape(Ynp, [-1])], axis=1)
npointsnp = np.shape(inputsnp)[0]

pool = multiprocessing.Pool()
neighbornp=pool.map(compute_neighbors, range(npointsnp))
neighbornp=np.reshape(neighbornp,(npointsnp,29))
pool.close()
pool.join()
damagenp = damage.numpy()

loss_plot = []
F_BC0 = []
F_BC1 = []

pool = multiprocessing.Pool()
Stiffnessnp=pool.map(compute_stiffness, range(2*npointsnp))
pool.close()
pool.join()
damagenp = damage.numpy()
Stiffnessnp=np.reshape(Stiffnessnp,(2*npointsnp,2*npointsnp))
RHS1np, Stiffnessnp = update_RHS_Stiff(npointsnp, dxdy, inputsnp, u_BC_0, x_max, x_min, Stiffnessnp)
RHS1 = tf.convert_to_tensor(RHS1np)
Stiffness = tf.convert_to_tensor(Stiffnessnp)
model = init_model(npoints)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(1.e-5, decay_steps=200, decay_rate=0.8, staircase=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
loss=10
epoch=0
try:
    uxy_linearized = tf.linalg.solve(Stiffness, -RHS1)
except:
    StiffnessSP=csc_matrix(Stiffnessnp)
    uxy_linearized, exitCode = cg(StiffnessSP, -RHS1np, tol=1e-9)
    uxy_linearized = tf.convert_to_tensor(uxy_linearized)
uxy_linearized = tf.reshape(uxy_linearized , [tf.shape(inputs)[0],2])
while loss > 1e-2:
    loss = train_step(inputs, inputs2, model, optimizer, id_innerMPs, id_BC0, id_BC1, id_damage, bond_c, Distance, eff_len, bond_dirX, bond_dirY, volume, damage, epoch, u_BC_0, uxy_linearized)
    loss_plot.append(loss)
    # if epoch % 1000 == 0:
    #     output = tf.transpose(model(tf.transpose(inputs2)))
    #     output = tf.reshape(output , [tf.shape(inputs)[0],2])
    #     plt.scatter(inputs[:, 0:1]+output[:, 0:1], inputs[:, 1:2]+output[:, 1:2], s=10, c=output[:, 0:1], marker='s', cmap='jet')
    #     plt.colorbar()
    #     plt.axis('equal')
    #     filename = 'plots_U1_'+str(epoch)+'.png'
    #     plt.savefig(filename)
    #     plt.close()
    epoch += 1
model.save_weights('pretrained_weights_PDINN.h5')
output = tf.transpose(model(tf.transpose(inputs2)))
output = tf.reshape(output , [tf.shape(inputs)[0],2])
Distance_denom = tf.where(tf.equal(Distance,0), tf.constant(1., dtype=tf.float64), Distance)
new_coords = output + inputs[:, 0:2]
x_new, y_new = new_coords[:, 0:1], new_coords[:, 1:2]
Xs=tf.tile(x_new, [1,tf.shape(inputs)[0]])
Ys=tf.tile(y_new, [1,tf.shape(inputs)[0]])
Distance_new = tf.sqrt(tf.square(Xs-tf.transpose(Xs))+tf.square(Ys-tf.transpose(Ys)))
stretch = (Distance_new-Distance)/Distance_denom
stretch = tf.where(tf.equal(eff_len,0.),tf.constant(0., dtype=tf.float64),stretch)
damage = tf.where(tf.greater(stretch,critical_stretch),tf.constant(1.,'float64'),damage)
damage = damage * id_non_Damage
damage = tf.where(tf.equal(damage,tf.transpose(damage)),damage,tf.constant(0., dtype=tf.float64))
netDamage = tf.reduce_sum(damage, axis=1) / (tf.reduce_sum(tf.where(tf.equal((eff_len),0.),tf.constant(0.,'float64'),tf.constant(1.,'float64')), axis=1)-1.)
plt.scatter(inputs[:, 0:1], inputs[:, 1:2], s=10, c=netDamage, marker='s', cmap='jet')
plt.colorbar()
plt.axis('equal')
filename = 'plot_netDamage.png'
plt.savefig(filename)
plt.close()
forceX = bond_c * volume**2 * tf.reduce_sum(stretch * bond_dirX * (eff_len) * (1.-(damage)), axis=1, keepdims=True)
forceY = bond_c * volume**2 * tf.reduce_sum(stretch * bond_dirY * (eff_len) * (1.-(damage)), axis=1, keepdims=True)
force=tf.sqrt(tf.reduce_sum(tf.square(tf.reshape(tf.stack([forceX,forceY], axis=1),[tf.shape(forceX)[0],2])),axis=1))
force=tf.reshape(force,[-1,1])
F_BC0.append(tf.reduce_sum(tf.gather(force,id_BC0)))
F_BC1.append(tf.reduce_sum(tf.gather(force,id_BC1)))
np.savetxt('F_BC0.txt', tf.convert_to_tensor(F_BC0, dtype=tf.float32).numpy())
np.savetxt('F_BC1.txt', tf.convert_to_tensor(F_BC1, dtype=tf.float32).numpy())
plt.scatter(inputs[:, 0:1]+output[:, 0:1], inputs[:, 1:2]+output[:, 1:2], s=10, c=output[:, 0:1], marker='s', cmap='jet')
plt.colorbar()
plt.axis('equal')
filename = 'Final_U1.png'
plt.savefig(filename)
plt.close()
# filename = 'netDamage.txt'
# np.savetxt(filename, netDamage.numpy())
loss_plot=tf.convert_to_tensor(loss_plot, dtype=tf.float32)
plt.plot(loss_plot[1:])
plt.ylim([0, 100])
plt.savefig('plot_loss.png')
plt.close()