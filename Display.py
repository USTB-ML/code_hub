### 一种显示数据集，并以label给每个图像命名的函数
### images_list 数组形图像列表
### labels_list 标签列表
def display(images_list,labels_list):
    for i in range(len(images_list)):
        plt.subplot(3,4,1+i).set_title(str(labels_list[i])),plt.imshow(images_list[i])
    plt.show()
