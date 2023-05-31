#!/usr/bin/env python
from utils.loading_utils import load_model, get_device
import numpy as np
import argparse
from utils.inference_utils import events_to_voxel_grid_pytorch
from scripts.image_reconstructor import ImageReconstructor
from options.inference_options import set_inference_options
import rospy
from sensor_msgs.msg import Image
from dvs_msgs.msg import EventArray
from cv_bridge import CvBridge
import datetime

class E2VID_ROS:
    def __init__(self):
        
        rospy.init_node('e2vid')
        
        self.frame_pub = rospy.Publisher("/e2vid/image", Image, queue_size=1)

        self.width = 640
        self.height = 480
        self.event_window_size = 1000
        self.camera_link = "camera_link"
        self.event_window = np.ones((4, self.event_window_size))
        print('Sensor size: {} x {}'.format(self.width, self.height))
        parser = argparse.ArgumentParser(description='Evaluating a trained network')
        set_inference_options(parser)
        self.args = parser.parse_args()
        self.bridge = CvBridge()
        self.last_timestamp = 0

        # Load model
        self.model = load_model("pretrained/E2VID_lightweight.pth.tar")
        self.device = get_device(self.args.use_gpu)
        model = self.model.to(self.device)
        model.eval()
        
        self.reconstructor = ImageReconstructor(model, self.height, self.width, self.model.num_bins, self.args)
        
        # Subscribe to the topic that publishes the event array
        rospy.Subscriber("/dvs/events", EventArray, self.event_array_callback)

    def loop(self):
        #print('prueba2')
        # rate = rospy.Rate(1)
        # while not rospy.is_shutdown():
        # # ROS callbacks
        #     rate.sleep()

            #print('prueba3')

            # Call the publish_reconstructed_image function to publish the image
            #e2vid.publish_reconstructed_image()

        rospy.spin()
            

    # Callback function to receive event_array data
    def event_array_callback(self, data):
        # Process the event_array data and store it in self.event_window
        # Conversion ROS message to a numpy array
        # print(data)
        #current_time = datetime.datetime.now()
        #print("Current time:", current_time)
        event_array = np.array(data.events)
        #print('Event Array Shape:', event_array.shape)
        #event_array = event_array.reshape((-1, 4))  # Reshape to 2D array
        
        #event_array = event_array[:self.event_window_size]

        #ts_list = []
        #x_list = []
        #y_list = []
        #polarity_list = []

        # Extract the values using NumPy array operations
        #ts_list = event_array['ts'].astype(np.float64)
        #x_list = event_array['x']
        #y_list = event_array['y']
        #polarity_list = event_array['polarity']

        # Truncate the event lists to the desired event_window_size
        #ts_list = ts_list[:self.event_window_size]
        #x_list = x_list[:self.event_window_size]
        #y_list = y_list[:self.event_window_size]
        #polarity_list = polarity_list[:self.event_window_size]


        ts_list = np.array([event.ts.to_sec() for event in event_array])
        x_list = np.array([event.x for event in event_array])
        y_list = np.array([event.y for event in event_array])
        polarity_list = np.array([event.polarity for event in event_array])
        #for i in range(0, len(event_array)):
        #    ts_list.append(event_array[i].ts.to_sec())
        #    x_list.append(event_array[i].x)
        #    y_list.append(event_array[i].y)
        #    polarity_list.append(event_array[i].polarity)
        #current_time2 = datetime.datetime.now()
        #print("Current time2:", current_time2 - current_time)
        self.event_window = np.ones((4, len(ts_list)), dtype=np.float64)
        #print(len(ts_list))

        # Update the self.event_window using the extracted information
        self.event_window[0, :] = ts_list
        self.event_window[1, :] = x_list
        self.event_window[2, :] = y_list
        self.event_window[3, :] = polarity_list
        last_timestamp = self.event_window[0, -1]
        self.last_timestamp = last_timestamp  # Update the instance variable
        #current_time3 = datetime.datetime.now()
        #print("Current time3:", current_time3 - current_time)
    # Callback function to publish reconstructed image
    #def publish_reconstructed_image(self):
        start_index = 0
        # Perform the reconstruction using self.event_window
        # and publish the reconstructed image using self.frame_pub
        event_tensor = events_to_voxel_grid_pytorch(self.event_window.transpose(),
                                                    num_bins=self.model.num_bins,
                                                    width=self.width,
                                                    height=self.height,
                                                    device=self.device)
        #print('prueba5')

        #current_time4 = datetime.datetime.now()
        #print("Current time4:", current_time4 - current_time)

        num_events_in_window = self.event_window.shape[0]

        out = self.reconstructor.update_reconstruction(event_tensor, start_index + num_events_in_window, self.last_timestamp)

        #current_time5 = datetime.datetime.now()
        #print("Current time5:", current_time5 - current_time)

        reconstructed_image = self.bridge.cv2_to_imgmsg(out, encoding="passthrough")
        reconstructed_image.header.stamp = rospy.Time(self.last_timestamp)
        reconstructed_image.header.frame_id = self.camera_link
        self.frame_pub.publish(reconstructed_image)
        start_index += num_events_in_window  

        #current_time6 = datetime.datetime.now()
        #print("Current time6:", current_time6 - current_time)

        #print('prueba6')


if __name__ == "__main__":
    #print('Empecemos')
    e2vid = E2VID_ROS()
    e2vid.loop()