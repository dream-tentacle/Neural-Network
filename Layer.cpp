#ifndef LAYER
#define LAYER
#ifndef IOSTREAM
#define IOSTREAM
#include<iostream>
#endif
class Layer
{
protected:
	int in_features, out_features;
	Layer* input, * output;
	double* input_data,* output_data;
	double cut(double number, double _min, double _max) {
		if (number < _min)return _min;
		if (number > _max)return _max;
		return number;
	}
public:
	Layer(int in_features, int out_features) :
		in_features(in_features), out_features(out_features) { 
		input = nullptr;
		output = nullptr;
		input_data = nullptr;
		output_data = nullptr;
	}
	virtual void random_originate(double min, double max) {
		if (output != nullptr) {
			output->random_originate(min, max);
		}
	}
	virtual void forward_once() {
		if (output != nullptr) {
			output->set_input_pointer(output_data);
			output->forward_once();
		}
	}
	virtual void learn(double learning_rate) {
		if (output != nullptr) {
			output->learn(learning_rate);
		}
	}
	virtual void backward_once(double* loss_gradient) {
		puts("ERRER: A layer does not have the function backward_once()!");
		exit(-1);
	}
	virtual void zero_gradient() {
		if (output != nullptr) {
			output->zero_gradient();
		}
	}
	void connect(Layer* next_layer) {
		///连接下一个层
		if (next_layer->in_features != out_features) {
			puts("ERRER: Two layers does not match!");
			exit(-1);
		}
		output = next_layer;
		next_layer->input = this;
	}
	void set_input_pointer(double* data) {
		input_data = data;
	}
};
class Input_layer: public Layer
{
protected:
	double learning_rate;
public:
	Input_layer(int input_size) :Layer(0, input_size) {
		//将层的输出设为神经网络输入，方便连接
		learning_rate = 0.01;
	}
	Input_layer(int input_size, double lr):Layer(0, input_size) {
		//将层的输出设为神经网络输入，方便连接
		learning_rate = lr;
	}
	void forward_once(double* Input_data) {
		if (output == nullptr) {
			puts("网络连接错误");
			return;
		}
		output->set_input_pointer(Input_data);
		output->forward_once();
	}
	void zero_gradient() {
		output->zero_gradient();
	}
	void backward_once(double* loss_gradient) {
		return;
	}
};
class Output_layer : public Layer
{
public:
	Output_layer(int input_size, int output_size) :Layer(input_size, output_size) {
		//将层的输入设为神经网络输出，方便连接
		input_data = new double[output_size];
	}
	virtual void forward_once() {
	}
	void print_output() {
		puts("network output results:");
		for (int i = 0; i < out_features; i++) {
			printf("%f%c", output_data[i], i + 1 == out_features ? '\n' : ' ');
		}
		puts("end");
	}
};
class Hidden_layer : public Layer
{
public:
	Hidden_layer(int in_features, int out_features) :Layer(in_features, out_features) {}
};
class L2_output_layer :public Output_layer
{
public:
	double* loss,* loss_gradient;
	L2_output_layer(int output_size) :Output_layer(output_size, output_size) {
		loss = new double[output_size];
		loss_gradient = new double[output_size];
		output_data = input_data;
	}
	void backward_once(double* true_labels) {
		for (int i = 0; i < in_features; i++)
		{
			loss_gradient[i] = 2 * (input_data[i] - true_labels[i]);
			loss[i] = (input_data[i] - true_labels[i]) * (input_data[i] - true_labels[i]);
		}
		input->backward_once(loss_gradient);
	}
	double loss_sum() {
		double re = 0;
		for (int i = 0; i < in_features; i++)
		{
			re += loss[i];
		}
		return re;
	}
};
#endif // !LAYER
