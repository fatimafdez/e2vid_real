#!/usr/bin/env python
from utils.loading_utils import load_model, get_device
import numpy as np
import argparse
from utils.inference_utils import events_to_voxel_grid_pytorch
from image_reconstructor import ImageReconstructor
from options.inference_options import set_inference_options
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class E2VID_ROS:
    def __init__(self):
        print('asdf')
        rospy.init_node('e2vid')
        print('asdyhgfhgfjf')
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

        # Load model
        self.model = load_model("pretrained/E2VID.pth.tar")
        self.device = get_device(self.args.use_gpu)
        model = self.model.to(self.device)
        model.eval()

        self.reconstructor = ImageReconstructor(model, self.height, self.width, self.model.num_bins, self.args)


    def loop(self):
    start_index = 0
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        # ROS callbacks
        
        # Callback function to receive event_array data
        def event_array_callback(data):
            # Process the event_array data and store it in self.event_window
            # Conversion ROS message to a numpy array
            event_array = np.array(data.events)

            # Event_array info
            timestamps = event_array[:, 0]
            x_values = event_array[:, 1]
            y_values = event_array[:, 2]
            polarities = event_array[:, 3]

            # Update the self.event_window using the extracted information
            self.event_window[0, :] = timestamps
            self.event_window[1, :] = x_values
            self.event_window[2, :] = y_values
            self.event_window[3, :] = polarities

        # Callback function to publish reconstructed image
        def publish_reconstructed_image():
            # Perform the reconstruction using self.event_window
            # and publish the reconstructed image using self.frame_pub
            event_tensor = events_to_voxel_grid_pytorch(self.event_window.transpose(),
                                                            num_bins=self.model.num_bins,
                                                            width=self.width,
                                                            height=self.height,
                                                            device=self.device)

            num_events_in_window = self.event_window.shape[0]
            

            reconstructed_image = self.bridge.cv2_to_imgmsg(out, encoding="passthrough")
            reconstructed_image.header.stamp = rospy.Time(last_timestamp)
            reconstructed_image.header.frame_id = self.camera_link
            self.frame_pub.publish(reconstructed_image)
            out = self.reconstructor.update_reconstruction(event_tensor, start_index + num_events_in_window, last_timestamp)




        # Subscribe to the topic that publishes the event_array
        rospy.Subscriber("/event_array_topic", Event_Array, event_array_callback)

        # Call the publish_reconstructed_image function to publish the image
        publish_reconstructed_image()

        start_index += num_events_in_window
        rate.sleep()


if __name__ == "__main__":
    print('holahola')
    e2vid = E2VID_ROS()
    e2vid.loop()