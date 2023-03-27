# Tiny Taichi Ray Tracer

This project is a real-time ray tracer implemented in Taichi, a high-performance programming language for parallel computing on GPUs in Python. The ray tracer is capable of rendering a scene with reflections in real-time.

## Features

- Real-time rendering 
- Support for materials and lighting
- Interactive camera controls with live updates (to-do)
- Efficient parallelization on GPUs using Taichi

## Installation

To use this ray tracer, you will need to install Taichi. Taichi can be installed using pip:

```sh
pip install taichi
```

Once Taichi is installed, you can clone this repository:

```sh
git clone https://github.com/your-username/taichi-ray-tracer.git
```

## Usage

To run the ray tracer, you can use the following command:

```sh
python tiny-taichi-raytracer.py
```

This will open a window showing the rendered scene. You can use the arrow keys to control the camera and the mouse to change the camera's direction. The rendered image will update in real-time as you move the camera.

## Examples

Examples will be uploaded once this project has been completed. At the moment, the scene does not render correctly, likely due to a bug in the parallelization of recursive Ray Tracing.

## Contributing

If you would like to contribute to this project, feel free to open a pull request or an issue. I would welcome contributions and help of any kind, whether it's adding new features, fixing bugs, or improving performance.

## Credits

This project was inspired by the repository [Tiny Ray Tracer](https://github.com/ssloy/tinyraytracer) by [ssloy](https://github.com/ssloy), and uses his code for the fundamentals.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
