#导入minidom
from xml.dom import minidom
#备注 ：20220615 物体旋转也写进了xml文件中 graspit可以显示成功

def real_robot_value(sum):
    a1 = min(sum, 1.5708)
    a2 = max(sum-1.5708, 0)
    return (a1, a2)

def label_value(aa, is_sum=False):
    if is_sum:
        a1 = aa/1.8
    else:
        a1 = aa
    a2 = 0.8*a1
    return (a1,a2)

def robot_selection(dom, a, rs):
    if rs[0] == 21:
        if rs[1] == 'real':
            name_text_rd = dom.createTextNode('{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}'.format(
                                                str(a[0]), str(a[1]), str(a[2]), str(real_robot_value(a[3])[0]), str(real_robot_value(a[3])[1]),
                                                str(a[4]), str(a[5]), str(real_robot_value(a[6])[0]), str(real_robot_value(a[6])[1]),
                                                str(a[7]), str(a[8]), str(real_robot_value(a[9])[0]), str(real_robot_value(a[9])[1]),
                                                str(a[10]), str(a[11]), str(real_robot_value(a[12])[0]), str(real_robot_value(a[12])[1]),
                                                str(a[13]), str(a[14]), str(a[15]), str(a[16]), str(a[17])))
        elif rs[1] == 'pretrain_single':
            name_text_rd = dom.createTextNode('{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}'.format(
                                                str(a[0]), str(a[1]), str(a[2]), str(label_value(a[3])[0]), str(label_value(a[3])[1]),
                                                str(a[4]), str(a[5]), str(label_value(a[6])[0]), str(label_value(a[6])[1]),
                                                str(a[7]), str(a[8]), str(label_value(a[9])[0]), str(label_value(a[9])[1]),
                                                str(a[10]), str(a[11]), str(label_value(a[12])[0]), str(label_value(a[12])[1]),
                                                str(a[13]), str(a[14]), str(a[15]), str(a[16]), str(a[17])))
        elif rs[1] == 'pretrain_sum':
            name_text_rd = dom.createTextNode('{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}'.format(
                                                str(a[0]), str(a[1]), str(a[2]), str(label_value(a[3], True)[0]), str(label_value(a[3], True)[1]),
                                                str(a[4]), str(a[5]), str(label_value(a[6], True)[0]), str(label_value(a[6], True)[1]),
                                                str(a[7]), str(a[8]), str(label_value(a[9], True)[0]), str(label_value(a[9], True)[1]),
                                                str(a[10]), str(a[11]), str(label_value(a[12], True)[0]), str(label_value(a[12], True)[1]),
                                                str(a[13]), str(a[14]), str(a[15]), str(a[16]), str(a[17])))
        else:
            raise ValueError('rs information is wrong')
    else:
        if len(a) == 18:
            name_text_rd=dom.createTextNode('{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}'.format(str(a[0]), str(a[1]), str(a[2]), str(a[3]), str(a[4]), str(a[5]), str(a[6]), str(a[7]), str(a[8]), str(a[9]), str(a[10]), str(a[11]), str(a[12]), str(a[13]), str(a[14]), str(a[15]), str(a[16]), str(a[17])))
        elif len(a) == 17:
            name_text_rd=dom.createTextNode('{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}'.format(str(a[0]), str(a[1]), str(a[2]), str(a[3]), str(a[4]), str(a[5]), str(a[6]), str(a[7]), str(a[8]), str(a[9]), str(a[10]), str(a[11]), str(a[12]), str(a[13]), str(a[14]), str(a[15]), str(a[16])))
        else:
            raise ValueError('the len of a is wrong')
    return name_text_rd


def write_xml_new_data(category, obj_name, r, r_o, t, a, path, mode='train', rs=(21, 'real')):
    dom=minidom.Document()
    world=dom.createElement('world')
    dom.appendChild(world)
    #  graspableBody
    graspableBody = dom.createElement('graspableBody')
    world.appendChild(graspableBody)

    filename = dom.createElement('filename')
    graspableBody.appendChild(filename)
    if mode=='new':
        name_text=dom.createTextNode('/hjl_data/{}/{}.xml'.format(str(category),str(obj_name[:-4])))
    else:
        name_text=dom.createTextNode('/hjl_data/{}/{}.xml'.format(str(category), str(obj_name[:-4])))
    filename.appendChild(name_text)

    transform = dom.createElement('transform')
    graspableBody.appendChild(transform)

    fullTransform = dom.createElement('fullTransform')
    transform.appendChild(fullTransform)
    # fullTransform_text=dom.createTextNode('(+1 +0 +0 +0)[+0 +0 +0]')
    fullTransform_text = dom.createTextNode('({} {} {} {})[+0 +0 +0]'.format(str(r_o[0]), str(r_o[1]), str(r_o[2]), str(r_o[3])))
    fullTransform.appendChild(fullTransform_text)

    # robot
    robot = dom.createElement('robot')
    world.appendChild(robot)

    filename_r = dom.createElement('filename')
    robot.appendChild(filename_r)
    # name_text_r=dom.createTextNode('/home/lm/graspit/models/robots/ShadowHand/ShadowHandSimple.xml')
    name_text_r=dom.createTextNode('/home/lm/graspit/models/robots/ShadowHand_gai/our_hand.xml')
    filename_r.appendChild(name_text_r)

    dofValues = dom.createElement('dofValues')
    robot.appendChild(dofValues)
    name_text_rd = robot_selection(dom, a, rs)
    dofValues.appendChild(name_text_rd)

    transform_r = dom.createElement('transform')
    robot.appendChild(transform_r)
    fullTransform_r = dom.createElement('fullTransform')
    transform_r.appendChild(fullTransform_r)
    fullTransform_text_r=dom.createTextNode('({} {} {} {})[{} {} {}]'.format(str(r[0]), str(r[1]), str(r[2]), str(r[3]), str(t[0]), str(t[1]), str(t[2])))
    fullTransform_r.appendChild(fullTransform_text_r)

    # camera
    camera = dom.createElement('camera')
    world.appendChild(camera)

    position = dom.createElement('position')
    camera.appendChild(position)
    name_text_c_1=dom.createTextNode('-2.21912 -6.21883 +479.86')
    position.appendChild(name_text_c_1)

    orientation = dom.createElement('orientation')
    camera.appendChild(orientation)
    name_text_c_2=dom.createTextNode('+0 +0 +0 +1')
    orientation.appendChild(name_text_c_2)

    focalDistance = dom.createElement('focalDistance')
    camera.appendChild(focalDistance)
    name_text_c_3=dom.createTextNode('+300.068')
    focalDistance.appendChild(name_text_c_3)



    try:
        with open(path,'w') as fh:
            # 4.writexml()第一个参数是目标文件对象，第二个参数是根节点的缩进格式，第三个参数是其他子节点的缩进格式，
            # 第四个参数制定了换行格式，第五个参数制定了xml内容的编码。
            dom.writexml(fh, indent='', addindent='\t', newl='\n', encoding='utf-8')
            # print('写入xml OK!')
    except Exception as err:
        print('错误信息：{0}'.format(err))
