# SwarmSpot

Tool for stitching inaccurate monocular depth estimates of scenes from multiple unknown perspectives into a high-fidelity depth map. The initial depth estimates are computed using the ZoeDepth transformer network, and then aligned with the base image using an initial pose garnered via OpenCV ORB feature tracking, followed by modified Iterative Closest Point registration. YOLO is used to detect target points in the base image, and the depths of those target points are averaged with those of the projected additional images to increase the accuracy of the estimates. The tool is tested for accuracy on a variety of images obtained in Unity simulation. SwarmSpot has the potential to be useful for depth perception in reconaissance with lightweight drones where position cannot necessarily be trusted.

More information can be found in our presentation here: https://docs.google.com/presentation/d/1BHZbK1TxwUPi73XMtYkWToUfc78CDQbsohJOiuU6FjA/edit?usp=sharing
