# -*- coding:UTF-8 -*-
#!/usr/bin/env python

import rospy
#from后边是自己的包.msg，也就是自己包的msg文件夹下，test是我的msg文件名test.msg
from test_msgs.msg import Test



def talker():
    pub = rospy.Publisher('chatter', Test, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        test=Test()
   	#temp为到时候要用于传输的信息
        test.vel = 21.0
        test.name = "hello"
        test.data = [10.0]
        test.pose.position.x = 1
        test.pose.position.y = 2
        test.pose.position.z = 3

        test.pose.orientation.x = 0.1
        test.pose.orientation.y = 0.2
        test.pose.orientation.z = 0.3
        test.pose.orientation.w = 0.4

        #这里test就像构造函数般使用，若有多个msg，那么写成test(a,b,c,d...)
        #rospy.loginfo(Test.name=temp)
        print(test)
        pub.publish(test)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass

