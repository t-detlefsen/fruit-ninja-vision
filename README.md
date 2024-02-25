# FruitNinja Vision
Detect the position and velocity of colored balls and predict where and when they will intersect with the ground plane.

## Installation
<ins>Create a ROS Workspace</ins>\
Refer to [the ROS tutorials](http://wiki.ros.org/ROS/Tutorials/InstallingandConfiguringROSEnvironment) for creating a new workspace.

<ins>Setup the Package</ins>\
Navigate to the `src` folder of your workspace and clone this repository.

```
cd ~/catkin_ws/src
git clone git@github.com:t-detlefsen/fruit-ninja-vision.git
```

Then, make `track-fruit.py`` an executable.

```
chmod +x ~/catkin_ws/src/fruit-ninja-vision/scripts/track-fruit.py
```

<ins>Make the Workspace</ins>\
Navigate to the root of your workspace and run `catkin_make`. Then source the workspace

```
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

## Usage
While there are RGB and depth images being published, run `track-fruit.py` by running:

```
rosrun fruit-ninja-vision track-fruit.py
```