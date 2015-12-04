from setuptools import setup

setup(
	name='neuralnetwork',
	version='0.1',
	description='A simple, small neural network implementation.',
	long_description='This project is a gist () made into a first-class citizen so it can be imported into another project.',
	classifiers=[
		'License :: OSI Approved :: MIT License',
		'Programming Language :: Python :: 2.7',
		'Topic :: Machine Learning :: Neural Network',
	],
	keywords='machine learning ML artificial intelligence AI neural network',
	url='http://github.com/josephcatrambone/pyneuralnetwork',
	author='Joseph Catrambone',
	author_email='me@josephcatrambone.com',
	license='MIT',
	packages=['neuralnetwork'],
	install_requires=['numpy'],
	test_suite='nose.collector',
	tests_require=['nose', 'nose-cover3'],
	zip_safe=False)
