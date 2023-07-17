#ifndef CNN
#define CNN
#include<cstring>
#include "Layer.cpp"
class Convolutional_neural_network : public Hidden_layer
{
protected:
	int input_height, input_width, input_channel;//The input picture's size
	int kernel_height, kernel_width;//The functioning area of one kernel
	int kernel_number;
	int step_size;
	double**** W = nullptr;
	double* b;
	double**** W_gradient = nullptr;
	double* b_gradient = nullptr;
public:
	Convolutional_neural_network(int input_height, int input_width, int input_channel,
		int kernel_height, int kernel_width, int kernel_number, int step_size) :
		input_height(input_height), input_width(input_width),
		input_channel(input_channel), kernel_height(kernel_height),
		kernel_width(kernel_width), step_size(step_size), kernel_number(kernel_number),
		Hidden_layer(input_height* input_width* input_channel, 
			(input_height - kernel_height + 1) / step_size *
			(input_width - kernel_width + 1) / step_size
			* kernel_number) {
		W = new double*** [kernel_number];
		b = new double[kernel_number];
		W_gradient = new double*** [kernel_number];
		b_gradient = new double[kernel_number];
		for (int i = 0; i < kernel_number; i++){
			W[i] = new double** [input_channel];
			for (int j = 0; j < input_channel; j++){
				W[i][j] = new double* [kernel_height];
				for (int k = 0; k < kernel_height; k++){
					W[i][j][k] = new double[kernel_width];
					for (int l = 0; l < kernel_width; l++){
						W[i][j][k][l] = 0;
;					}
				}
			}
		}
		for (int i = 0; i < kernel_number; i++)
			b[i] = 0;
		for (int i = 0; i < kernel_number; i++) {
			W_gradient[i] = new double** [input_channel];
			for (int j = 0; j < input_channel; j++) {
				W_gradient[i][j] = new double* [kernel_height];
				for (int k = 0; k < kernel_height; k++) {
					W_gradient[i][j][k] = new double[kernel_width];
					for (int l = 0; l < kernel_width; l++) {
						W_gradient[i][j][k][l] = 0;
					}
				}
			}
		}
		for (int i = 0; i < kernel_number; i++)
			b_gradient[i] = 0;
	}
	void random_originate(double min, double max) {
		srand(time(0));
		for (int i = 0; i < kernel_number; i++)
			for (int j = 0; j < input_channel; j++)
				for (int k = 0; k < kernel_height; k++) 
					for (int l = 0; l < kernel_width; l++)
						W[i][j][k][l] = (max - min) * 1.0 * rand() / RAND_MAX + min;
		for (int j = 0; j < kernel_number; j++)
			b[j] = (max - min) * 1.0 * rand() / RAND_MAX + min;
		if (output != nullptr) {
			output->random_originate(min, max);
		}
	}
	void forward_once() {
		if (output_data == nullptr)output_data = new double[out_features];
		const int n = kernel_number, x = input_height - kernel_height + 1,
			y = input_width - kernel_width + 1;
		int cnt = 0;
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < x; j+=step_size)
			{
				for (int k = 0; k < y; k+=step_size)
				{
					const int tmp = cnt++;
					output_data[tmp] = 0;
					for (int ch = 0; ch < input_channel; ch++)
					{
						for (int input_y = 0; input_y < kernel_height; input_y++)
						{
							for (int input_x = 0; input_x < kernel_width; input_x++)
							{
								
								output_data[tmp] += input_data[ch * input_height * input_width
									+ (input_y + j) * input_width + input_x + k]
									* W[i][ch][input_y][input_x];
							}
						}
					}
					output_data[tmp] += b[i];
				}
			}
		}
		output->set_input_pointer(output_data);
		output->forward_once();
	}
	void backward_once(double* loss_gradient) {
		const int n = kernel_number, x = input_height - kernel_height + 1,
			y = input_width - kernel_width + 1;
		int cnt = 0;
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < x; j += step_size)
			{
				for (int k = 0; k < y; k += step_size)
				{
					const int tmp = cnt++;
					for (int ch = 0; ch < input_channel; ch++)
					{
						for (int input_y = 0; input_y < kernel_height; input_y++)
						{
							for (int input_x = 0; input_x < kernel_width; input_x++)
							{
								W_gradient[i][ch][input_y][input_x] +=
									input_data[ch * input_height * input_width
									+ (input_y + j) * input_width + input_x + k]
									* loss_gradient[tmp];
							}
						}
					}
					b_gradient[i] += loss_gradient[tmp];
				}
			}
		}
		for (int i = 0; i < kernel_number; i++) {
			for (int ch = 0; ch < input_channel; ch++)
				for (int input_y = 0; input_y < kernel_height; input_y++)
					for (int input_x = 0; input_x < kernel_width; input_x++) {
						W_gradient[i][ch][input_y][input_x] =
							cut(W_gradient[i][ch][input_y][input_x], -1.0, 1.0);
					}
			b_gradient[i] = cut(b_gradient[i], -1.0, 1.0);
		}

		double* next_loss_gradient = new double[in_features];
		for (int i = 0; i < in_features; i++)
			next_loss_gradient[i] = 0;
		cnt = 0;
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < x; j += step_size)
			{
				for (int k = 0; k < y; k += step_size)
				{
					const int tmp = cnt++;
					for (int ch = 0; ch < input_channel; ch++)
					{
						for (int input_y = 0; input_y < kernel_height; input_y++)
						{
							for (int input_x = 0; input_x < kernel_width; input_x++)
							{
								
								next_loss_gradient[ch * input_height * input_width
									+ (input_y + j) * input_width + input_x + k]
									+= loss_gradient[tmp]
									* W[i][ch][input_y][input_x];
							}
						}
					}
				}
			}
		}
		input->backward_once(next_loss_gradient);
		delete[]next_loss_gradient;
		//next_loss_gradient = nullptr;
	}
	void print_W() {
		for (int i = 0; i < kernel_number; i++)
			for (int ch = 0; ch < input_channel; ch++)
				for (int input_y = 0; input_y < kernel_height; input_y++)
					for (int input_x = 0; input_x < kernel_width; input_x++)
						printf("%d %d %d %d %f\n", i, ch, input_y, input_x, W[i][ch][input_y][input_x]);
		puts("conmpleted!\n");
	}
	void zero_gradient() {
		for (int i = 0; i < kernel_number; i++)
			for (int ch = 0; ch < input_channel; ch++)
				for (int input_y = 0; input_y < kernel_height; input_y++)
					for (int input_x = 0; input_x < kernel_width; input_x++) {
						W_gradient[i][ch][input_y][input_x] = 0;
					}
		for (int i = 0; i < kernel_number; i++)
			b_gradient[i] = 0;
		output->zero_gradient();
	}
	void learn(double learning_rate) {
		for (int i = 0; i < kernel_number; i++)
			for (int ch = 0; ch < input_channel; ch++)
				for (int input_y = 0; input_y < kernel_height; input_y++)
					for (int input_x = 0; input_x < kernel_width; input_x++)
						W[i][ch][input_y][input_x] -= learning_rate
						* W_gradient[i][ch][input_y][input_x];
		for (int i = 0; i < kernel_number; i++)
		{
			if (!isnan(b[i] - b_gradient[i] * learning_rate))
				b[i] = b[i] - b_gradient[i] * learning_rate;
		}
		if (output != nullptr) {
			output->learn(learning_rate);
		}
	}
	void print_gradient() {
		for (int i = 0; i < kernel_number; i++) {
			printf("[");
			for (int ch = 0; ch < input_channel; ch++) {
				printf("[");
				for (int input_y = 0; input_y < kernel_height; input_y++) {
					printf("[");
					for (int input_x = 0; input_x < kernel_width; input_x++) {
						printf("%f ", W_gradient[i][ch][input_y][input_x]);
					}
					printf("],");
				}
				printf("],");
			}
			printf("],");
		}
		printf("[");
		for (int i = 0; i < kernel_number; i++)
		{
			printf("%f%c", b_gradient[i], i == kernel_number - 1 ? ']' : ' ');
		}
	}
};
class Max_pool : public Hidden_layer
{
protected:
	int input_height, input_width, input_channel;//The input picture's size
	int pool_len;
public:
	Max_pool(int input_height, int input_width, int input_channel, int pool_len) :
		input_height(input_height), input_width(input_width),
		input_channel(input_channel),
		pool_len(pool_len), Hidden_layer(input_height* input_width* input_channel,
			(input_height / pool_len)* (input_width / pool_len)* input_channel) {}
	void forward_once() {
		if (output_data == nullptr)output_data = new double[out_features];
		int cnt = 0;
		for (int ch = 0; ch < input_channel; ch++) {
			for (int i = 0; i < input_height - pool_len + 1; i += pool_len) {
				for (int j = 0; j < input_width - pool_len + 1; j += pool_len) {
					const int ch_offset = ch * input_height * input_width;
					const int tmp = cnt++;
					output_data[tmp] = input_data[ch_offset + i * input_width + j];
					for (int a = i; a < i + pool_len; a++) {
						for (int b = j; b < j + pool_len; b++) {
							if (output_data[tmp] <
								input_data[ch_offset + a * input_width + b])
								output_data[tmp] =
								input_data[ch_offset + a * input_width + b];
						}
					}
				}
			}
		}
		output->set_input_pointer(output_data);
		output->forward_once();
	}
	void backward_once(double* loss) {
		double* next_loss = new double[in_features];
		for (int i = 0; i < in_features; i++)
			next_loss[i] = 0;
		int cnt = 0;
		for (int ch = 0; ch < input_channel; ch++) {
			for (int i = 0; i < input_height - pool_len + 1; i += pool_len) {
				for (int j = 0; j < input_width - pool_len + 1; j += pool_len) {
					const int ch_offset = ch * input_height * input_width;
					int choose_max = ch_offset + i * input_width + j;
					for (int a = i; a < i + pool_len; a++) {
						for (int b = j; b < j + pool_len; b++) {
							if (output_data[choose_max] <
								input_data[ch_offset + a * input_width + b])
								choose_max = ch_offset + a * input_width + b;
						}
					}
					next_loss[choose_max] += loss[cnt++];
				}
			}
		}
		input->backward_once(next_loss);
		delete[]next_loss;
		//next_loss = nullptr;
	}

};
#endif
