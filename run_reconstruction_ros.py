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

class E2VID_ROS:
    def __init__(self):
        
        rospy.init_node('e2vid')
        
        self.frame_pub = rospy.Publisher("/e2vid/image", Image, queue_size=1)
        self.width = 640
        self.height = 480
        self.event_window_size = 30000
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
        print('prueba1')


    def loop(self):
        print('prueba2')
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
        # ROS callbacks
            rate.sleep()
            print('prueba3')

            # Call the publish_reconstructed_image function to publish the image
            e2vid.publish_reconstructed_image()
            

    # Callback function to receive event_array data
    def event_array_callback(self, data):
            # Process the event_array data and store it in self.event_window
            # Conversion ROS message to a numpy array
            #print(data)
        event_array = np.array(data.events)
        #print('Event Array Shape:', event_array.shape)
        #event_array = event_array.reshape((-1, 4))  # Reshape to 2D array
        ts_list = []
        x_list = []
        y_list = []
        polarity_list = []

        #print(event_array)
        #print(event_array[0])

        for i in range(0, len(event_array)):
            ts_list.append(event_array[i].ts.to_sec())
            x_list.append(event_array[i].x)
            y_list.append(event_array[i].y)
            polarity_list.append(event_array[i].polarity)

        #    ts_list[i] = event_array(0)
        #    x_list[i] = event_array(1)
        #    y_list[i] = event_array(2)
        #    polarity_list[i] = event_array(3)
            
     
        self.event_window = np.ones((4, len(ts_list)), dtype=np.float64)
        # Update the self.event_window using the extracted information
        self.event_window[0, :] = ts_list
        self.event_window[1, :] = x_list
        self.event_window[2, :] = y_list
        self.event_window[3, :] = polarity_list
        last_timestamp = self.event_window[0, -1]
        self.last_timestamp = last_timestamp  # Update the instance variable

        #print('prueba6')
         
    # Callback function to publish reconstructed image
    def publish_reconstructed_image(self):
        start_index = 0
        # Perform the reconstruction using self.event_window
        # and publish the reconstructed image using self.frame_pub
        event_tensor = events_to_voxel_grid_pytorch(self.event_window.transpose(),
                                                    num_bins=self.model.num_bins,
                                                    width=self.width,
                                                    height=self.height,
                                                    device=self.device)
        print('prueba7')
        
        num_events_in_window = self.event_window.shape[0]
            
        #print('out type:', type(out))
        #print('out shape:', out.shape)

        # event_tensor = np.array(event_tensor)
        out = self.reconstructor.update_reconstruction(event_tensor, start_index + num_events_in_window, self.last_timestamp)
        # if isinstance(out, torch.Tensor):
        # out = out.numpy()
        # print(out)
        # self.reconstructor.
        reconstructed_image = self.bridge.cv2_to_imgmsg(out, encoding="passthrough")
        reconstructed_image.header.stamp = rospy.Time(self.last_timestamp)
        reconstructed_image.header.frame_id = self.camera_link
        self.frame_pub.publish(reconstructed_image)
        start_index += num_events_in_window  
        print('prueba8')


if __name__ == "__main__":
    print('Empecemos')
    e2vid = E2VID_ROS()
    e2vid.loop()