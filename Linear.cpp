#ifndef LINEAR
#define LINEAR
#include "Layer.cpp"
#ifndef RAND
#define RAND
#include<ctime>
#endif
class Linear: public Hidden_layer
{
protected:
	double** W=nullptr;//二维指针。可以动态建立数组
	double* b;
	double** W_gradient = nullptr;
	double* b_gradient = nullptr;
public:
	Linear(int in_features, int out_features) :
		Hidden_layer(in_features, out_features){
		W = new double* [out_features];
		for (int i = 0; i < out_features; i++)
		{
			W[i] = new double[in_features];
			for (int j = 0; j < in_features; j++)
			{
				W[i][j] = 0;
			}
		}
		b = new double[out_features];
		for (int i = 0; i < out_features; i++)
		{
			b[i] = 0;
		}
	}
	void set_W(double** data) {
		for (int i = 0; i < out_features; i++)
		{
			for (int j = 0; j < in_features; j++)
			{
				W[i][j] = data[i][j];
			}
		}
	}
	void random_originate(double min, double max) {
		srand(time(0));
		for (int i = 0; i < out_features; i++)
			for (int j = 0; j < in_features; j++)
				W[i][j] = (max - min) * 1.0 *rand() /RAND_MAX + min;
		for (int j = 0; j < out_features; j++)
			b[j] = (max - min) * 1.0 * rand() / RAND_MAX + min;
		if (output != nullptr) {
			output->random_originate(min, max);
		}
	}
	void forward_once() {
		if (output_data == nullptr)output_data = new double[out_features];
		for (int i = 0; i < out_features; i++) {
			output_data[i] = 0;
			for (int j = 0; j < in_features; j++) {
				output_data[i] += input_data[j] * W[i][j];
			}
			output_data[i] += b[i];
		}
		output->set_input_pointer(output_data);
		output->forward_once();
	}
	void backward_once(double* loss_gradient) {
		for (int i = 0; i < out_features; i++){
			for (int j = 0; j < in_features; j++)
			{
				W_gradient[i][j] += input_data[j] * loss_gradient[i];
			}
		}
		for (int i = 0; i < out_features; i++)
		{
			b_gradient[i] +=loss_gradient[i];
		}
		double* next_loss_gradient = new double[in_features];
		for (int i = 0; i < in_features; i++)
		{
			next_loss_gradient[i] = 0;
			for (int j = 0; j < out_features; j++)
			{
				next_loss_gradient[i] += loss_gradient[j] * W[j][i];
			}
		}
		input->backward_once(next_loss_gradient);
		delete []next_loss_gradient;
		//next_loss_gradient = nullptr;
	}	
	void print_gradient() {
		puts("W_gradient: ");
		for (int i = 0; i < out_features; i++)
		{
			for (int j = 0; j < in_features; j++)
			{
				printf("%f%c", W_gradient[i][j], j == in_features - 1 ? '\n' : ' ');
			}
		}
		for (int i = 0; i < out_features; i++)
		{
			printf("%f%c", b_gradient[i], i == in_features - 1 ? '\n' : ' ');
		}
		puts("end");
	}
	void print_value() {
		puts("value:");
		for (int i = 0; i < out_features; i++)
		{
			for (int j = 0; j < in_features; j++)
			{
				printf("%f%c", W[i][j], j == in_features - 1 ? '\n' : ' ');
			}
		}
		for (int i = 0; i < out_features; i++)
		{
			printf("%f%c", b[i], i == in_features - 1 ? '\n' : ' ');
		}
		puts("value end");
	}
	void zero_gradient() {
		if (W_gradient == nullptr) {
			W_gradient = new double* [out_features];
			for (int i = 0; i < out_features; i++) {
				W_gradient[i] = new double[in_features];
			}
		}
		if (b_gradient == nullptr) {
			b_gradient = new double[out_features];
		}
		for (int i = 0; i < out_features; i++)
		{
			for (int j = 0; j < in_features; j++)
			{
				W_gradient[i][j] = 0;
			}
		}
		for (int i = 0; i < out_features; i++)
		{
			b_gradient[i] = 0;
		}
		output->zero_gradient();
	}
	void learn(double learning_rate) {
		for (int i = 0; i < out_features; i++)
		{
			for (int j = 0; j < in_features; j++)
			{
				if (!isnan(W[i][j] - W_gradient[i][j] * learning_rate))
					W[i][j] = W[i][j] - W_gradient[i][j] * learning_rate;
			}
		}
		for (int i = 0; i < out_features; i++)
		{
			if(!isnan(b[i] - b_gradient[i] * learning_rate))
				b[i] = b[i] - b_gradient[i] * learning_rate;
		}
		if (output != nullptr) {
			output->learn(learning_rate);
		}
	}
};
#endif // !LINEAR
