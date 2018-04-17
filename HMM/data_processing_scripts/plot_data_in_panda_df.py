from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


pos_plot = None
ori_x = None
ori_y = None
ori_z = None
ori_w = None
fx = None
fy = None
fz = None
mx = None
my = None
mz = None


def init_plots():
    global pos_plot
    global ori_x
    global ori_y
    global ori_z
    global ori_w
    global fx
    global fy
    global fz
    global mx
    global my
    global mz
    global jp1, jp2, jp3, jp4, jp5, jp6, jp7

    fig = plt.figure()
    pos_plot = fig.add_subplot(111, projection='3d')

    fig = plt.figure()
    ori_x = fig.add_subplot(411)
    ori_y = fig.add_subplot(412)
    ori_z = fig.add_subplot(413)
    ori_w = fig.add_subplot(414)

    fig = plt.figure()
    fx = fig.add_subplot(231)
    fy = fig.add_subplot(232)
    fz = fig.add_subplot(233)
    mx = fig.add_subplot(234)
    my = fig.add_subplot(235)
    mz = fig.add_subplot(236)

    fig = plt.figure()
    jp1 = fig.add_subplot(241)
    jp2 = fig.add_subplot(242)
    jp3 = fig.add_subplot(243)
    jp4 = fig.add_subplot(244)
    jp5 = fig.add_subplot(245)
    jp6 = fig.add_subplot(246)
    jp7 = fig.add_subplot(247)

def plot_legend():
    global pos_plot
    global ori_x
    global ori_y
    global ori_z
    global ori_w
    global fx
    global fy
    global fz
    global mx
    global my
    global mz
    global jp1, jp2, jp3, jp4, jp5, jp6, jp7

    pos_plot.legend()
    ori_x.legend()
    ori_y.legend()
    ori_z.legend()
    ori_w.legend()
    fx.legend()
    fy.legend()
    fz.legend()
    mx.legend()
    my.legend()
    mz.legend()
    jp1.legend()
    jp2.legend()
    jp3.legend()
    jp4.legend()
    jp5.legend()
    jp6.legend()
    jp7.legend()


def plot_one_df(df, color, label):
    global pos_plot
    global ori_x
    global ori_y
    global ori_z
    global ori_w
    global fx
    global fy
    global fz
    global mx
    global my
    global mz
    global jp1, jp2, jp3, jp4, jp5, jp6, jp7
    

    if '.endpoint_state.pose.position.x' in df:
        pos_plot.plot(
            df['.endpoint_state.pose.position.x'].tolist(), 
            df['.endpoint_state.pose.position.y'].tolist(), 
            df['.endpoint_state.pose.position.z'].tolist(), 
            color=color,
            label=label
        )
        pos_plot.set_title("pos xyz")

        ori_x.plot(
            df.index.tolist(),
            df['.endpoint_state.pose.orientation.x'].tolist(), 
            color=color,
            label=label
        )
        ori_x.set_title("ori x")

        ori_y.plot(
            df.index.tolist(),
            df['.endpoint_state.pose.orientation.y'].tolist(), 
            color=color,
            label=label
        )
        ori_y.set_title("ori y")

        ori_z.plot(
            df.index.tolist(),
            df['.endpoint_state.pose.orientation.z'].tolist(), 
            color=color,
            label=label
        )
        ori_z.set_title("ori z")

        ori_w.plot(
            df.index.tolist(),
            df['.endpoint_state.pose.orientation.w'].tolist(), 
            color=color,
            label=label
        )
        ori_w.set_title("ori w")


    if '.wrench_stamped.wrench.force.x' in df:
        fx.plot(
            df.index.tolist(),
            df['.wrench_stamped.wrench.force.x'].tolist(), 
            color=color,
            label=label)
        fx.set_title("fx")

        fy.plot(
            df.index.tolist(),
            df['.wrench_stamped.wrench.force.y'].tolist(), 
            color=color,
            label=label)
        fy.set_title("fy")

        fz.plot(
            df.index.tolist(),
            df['.wrench_stamped.wrench.force.z'].tolist(), 
            color=color,
            label=label)
        fz.set_title("fz")

        mx.plot(
            df.index.tolist(),
            df['.wrench_stamped.wrench.torque.x'].tolist(), 
            color=color,
            label=label)
        mx.set_title("mx")

        my.plot(
            df.index.tolist(),
            df['.wrench_stamped.wrench.torque.y'].tolist(), 
            color=color,
            label=label)
        my.set_title("my")

        mz.plot(
            df.index.tolist(),
            df['.wrench_stamped.wrench.torque.z'].tolist(), 
            color=color,
            label=label)
        mz.set_title("mz")

    if '.CartesianWrench.wrench.force.x' in df:
        fx.plot(
            df.index.tolist(),
            df['.CartesianWrench.wrench.force.x'].tolist(), 
            color=color,
            label=label)
        fx.set_title("fx")

        fy.plot(
            df.index.tolist(),
            df['.CartesianWrench.wrench.force.y'].tolist(), 
            color=color,
            label=label)
        fy.set_title("fy")

        fz.plot(
            df.index.tolist(),
            df['.CartesianWrench.wrench.force.z'].tolist(), 
            color=color,
            label=label)
        fz.set_title("fz")

        mx.plot(
            df.index.tolist(),
            df['.CartesianWrench.wrench.torque.x'].tolist(), 
            color=color,
            label=label)
        mx.set_title("mx")

        my.plot(
            df.index.tolist(),
            df['.CartesianWrench.wrench.torque.y'].tolist(), 
            color=color,
            label=label)
        my.set_title("my")

        mz.plot(
            df.index.tolist(),
            df['.CartesianWrench.wrench.torque.z'].tolist(), 
            color=color,
            label=label)
        mz.set_title("mz")


    if '.joint_state.position.right_s0' in df:
        jp1.plot(
            df.index.tolist(),
            df['.joint_state.position.right_s0'].tolist(),
            color = color,
            label = label)
        jp1.set_title('jp1')
        
        jp2.plot(
            df.index.tolist(),
            df['.joint_state.position.right_s1'].tolist(),
            color = color,
            label = label)
        jp2.set_title('jp2')

        jp3.plot(
            df.index.tolist(),
            df['.joint_state.position.right_e0'].tolist(),
            color = color,
            label = label)
        jp3.set_title('jp3')

        jp4.plot(
            df.index.tolist(),
            df['.joint_state.position.right_e1'].tolist(),
            color = color,
            label = label)
        jp4.set_title('jp4')

        jp5.plot(
            df.index.tolist(),
            df['.joint_state.position.right_w0'].tolist(),
            color = color,
            label = label)
        jp5.set_title('jp5')

        jp6.plot(
            df.index.tolist(),
            df['.joint_state.position.right_w1'].tolist(),
            color = color,
            label = label)
        jp6.set_title('jp6')

        jp7.plot(
            df.index.tolist(),
            df['.joint_state.position.right_w2'].tolist(),
            color = color,
            label = label)
        jp7.set_title('jp7')

def show_plots():
    plt.tight_layout()
    plt.show()
