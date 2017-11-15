from PIL import Image # 从数据，生成image对象
for i in range(100):
    # i = 0
    img = X[i] # 'channels_first'
    img_R = img[0]
    img_G = img[1]
    img_B = img[2]
    R = Image.fromarray(img_R)
    G = Image.fromarray(img_G)
    B = Image.fromarray(img_B)
    pic = Image.merge(mode="RGB", bands=(R,G,B))
    name = label[i]
    pic.save("D:\\my_project\\Python_Project\\test\\NN\\cifar-10-python\\test\\"+name, "png")
