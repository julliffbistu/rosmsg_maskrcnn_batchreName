#!/usr/bin/env python
import rospy
from test_msgs.msg import Test
 
def callback(data):
    print("-----------",data.name)
    print("-----------",data.vel)
    print("-----------",data.pose.position.x)
    print("-----------",data.pose.position.y)
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
    
def listener():
 
    rospy.init_node('listener', anonymous=True)
 
    rospy.Subscriber("chatter", Test, callback)
 
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
 
if __name__ == '__main__':
    listener()
