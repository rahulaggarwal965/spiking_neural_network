#include "snn_internal.h"
#include <cstdlib>
#include <random>
#include <stdio.h>
#include <string.h>
#include <vector>

/* const f32 MV_MIN = -80.0f; */
/* const f32 MV_REST = -70.0f; */
/* const f32 MV_THRESH = -50.0f; */
/* const f32 MV_SPIKE = 90.0f; */
/* const f32 MV_DECAY = 5.0f; */

const f32 MV_MIN = -1;
const f32 MV_REST = 0;
const f32 MV_THRESH = 25;
const f32 MV_SPIKE = 5;
const f32 MV_DECAY = 0.25;

const u32 REFRACTORY_PERIOD = 3; // define in time units

#define RECORD_SPIKES 1

#if RECORD_SPIKES
#include <matplotlib-cpp/matplotlibcpp.h>

namespace plt = matplotlibcpp;
#endif

struct Neuron {
    f64 membrane_potential = MV_REST;
    f32 refractory_period = 0; // define in time units
};

struct Matrix {
    u32 rows, cols;
    f32 *data;
};

inline Matrix create_matrix(const u32 rows, const u32 cols) {
    Matrix r;
    r.rows = rows;
    r.cols = cols;
    /* r.data = (f32 *) calloc(rows * cols * sizeof(f32)); */
    r.data = (f32 *)calloc(rows * cols, sizeof(f32));
    return r;
}

inline void free_matrix(Matrix &m) { free(m.data); }

inline void print_matrix(const Matrix &m) {
    memory_index k = 0;
    printf("[ ");
    for (memory_index i = 0; i < m.cols; i++) {
        for (memory_index j = 0; j < m.rows; j++) {
            printf("%f, ", m.data[k++]);
        }
        printf("\n");
    }

    printf("]\n");
}

struct SNN {
    u32 num_neurons;
    Matrix weight_matrix;
    std::vector<Neuron> neurons;
    f32 current_time = 0;
    f32 *spike_train;
#if RECORD_SPIKES
    std::vector<f32> *spikes;
    std::vector<f32> *membrane_potentials;
#endif
};

void multiply_mv(const Matrix &m, const f32 *v, f32 *r) {
    memset(r, 0, m.rows * sizeof(f32));
    memory_index k = 0;
    for (memory_index j = 0; j < m.rows; j++) {
        for (memory_index i = 0; i < m.cols; i++) {
            r[j] += m.data[k++] * v[j];
        }
    }
}

void multiply_transpose_mv(const Matrix &m, const f32 *v, f32 *r) {
    memset(r, 0, m.rows * sizeof(f32));
    memory_index k = 0;
    for (memory_index j = 0; j < m.rows; j++) {
        for (memory_index i = 0; i < m.cols; i++) {
            r[i] += m.data[k++] * v[j];
        }
    }
}

void update(SNN &snn, f32 delta_time) {
    f32 state_change[snn.num_neurons];
    multiply_transpose_mv(snn.weight_matrix, snn.spike_train, state_change);
    memset(snn.spike_train, 0, snn.num_neurons * sizeof(f32));

    for (memory_index i = 0; i < snn.num_neurons; i++) {
        Neuron &neuron = snn.neurons[i];
#if RECORD_SPIKES
        snn.membrane_potentials[i].push_back(neuron.membrane_potential);
#endif
        if (neuron.refractory_period > 0) {

            neuron.membrane_potential = MV_REST;
            neuron.refractory_period -= delta_time;
        } else {
            if (neuron.membrane_potential < MV_MIN) {
                neuron.membrane_potential = MV_REST;
            } else {
                neuron.membrane_potential += state_change[i] - MV_DECAY;
            }
        }
        if (neuron.membrane_potential >= MV_THRESH) {
            snn.spike_train[i] = 1;
#if RECORD_SPIKES
            snn.spikes[i][(memory_index)(snn.current_time * 4)] = 1; // for graphing
#endif
            neuron.refractory_period = REFRACTORY_PERIOD;
            neuron.membrane_potential += MV_SPIKE;
        }
    }
    snn.current_time += delta_time;
}

inline SNN create_snn(const u32 num_neurons, bool randomize_synapses = true) {
    SNN snn;
    snn.num_neurons = num_neurons;
    snn.weight_matrix = create_matrix(num_neurons, num_neurons);
    snn.spike_train = (f32 *)calloc(num_neurons, sizeof(f32));
    f32 *weight_matrix_data = snn.weight_matrix.data;
    if (randomize_synapses) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<f32> distr(-1, 1);

        for (memory_index i = 0; i < snn.weight_matrix.rows; i++) {
            for (memory_index j = 0; j < snn.weight_matrix.cols; j++) {
                const memory_index index = i + j * snn.weight_matrix.cols;
                if (i == j || distr(gen) < 0.4)
                    weight_matrix_data[index] = 0;
                else
                    weight_matrix_data[index] = distr(gen);
            }
        }
    }
    snn.neurons.resize(num_neurons);

    return snn;
}

inline void free_snn(SNN &snn) {
    free_matrix(snn.weight_matrix);
    free(snn.spike_train);
}

template <typename T> void print_vector(const std::vector<T> &v) {
    printf("[");
    for (memory_index i = 0; i < v.size() - 1; i++) {
        std::cout << v[i] << ", ";
    }
    std::cout << v.back() << "]\n";
}

template <typename T> void print_buffer(T *buffer, memory_index size) {
    printf("[");
    for (memory_index i = 0; i < size - 1; i++) {
        std::cout << buffer[i] << ", ";
    }
    std::cout << buffer[size - 1] << "]\n";
}

int main(int argc, char **argv) {
    const f32 time_unit_limit = 500;
    const f32 delta_time = 0.25;
    const u32 num_neurons = 20;
    SNN snn = create_snn(num_neurons);
    print_matrix(snn.weight_matrix);

    // temporary random seeding of the neurons
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<f32> distr;
    for (memory_index i = 0; i < snn.num_neurons; i++) {
        Neuron &neuron = snn.neurons[i];
        if (distr(gen) < 0.3) {
            neuron.membrane_potential = MV_THRESH;
            snn.spike_train[i] = 1;
        }
    }
    print_buffer(snn.spike_train, snn.num_neurons);

#if RECORD_SPIKES
    std::vector<f32> time_units;
    std::vector<f32> spikes[num_neurons];
    std::vector<f32> membrane_potentials[num_neurons];
    time_units.reserve(time_unit_limit);
    for (f32 i = 0; i < time_unit_limit; i += delta_time) {
        time_units.push_back(i);
    }
    for (memory_index i = 0; i < num_neurons; i++) {
        spikes[i].resize(time_unit_limit / delta_time, 0);
        membrane_potentials[i].reserve(time_unit_limit);
    }
    snn.spikes = spikes;
    snn.membrane_potentials = membrane_potentials;

#endif

    bool continueRunning = true;
    std::uniform_real_distribution<f32> random_input(-1, 2);

    while (continueRunning) {

        // give neuron 0 some random input
        if (snn.neurons[0].refractory_period == 0) {
            snn.neurons[0].membrane_potential += random_input(gen);
        }
        if (snn.neurons[1].refractory_period == 0) {
            snn.neurons[1].membrane_potential += random_input(gen);
        }



        update(snn, delta_time);
        if (snn.current_time >= time_unit_limit) {
            continueRunning = false;
        }
    }

    free_snn(snn);

#if RECORD_SPIKES

    /* print_vector(time_units); */
    printf("Spike length: %zu\n", snn.spikes[0].size());
    /* print_vector(snn.spikes[0]); */
    printf("Membrane Potential length: %zu\n", snn.membrane_potentials[0].size());
    /* print_vector(snn.membrane_potentials[0]); */

    plt::suptitle("Neuron 0");
    plt::subplot(2, 1, 1);
    plt::ylim(MV_MIN - 1, MV_THRESH + MV_SPIKE + 10);
    plt::title("Neuron Voltage (mV)");
    for (memory_index i = 0; i < snn.num_neurons; i++) {
        /* plt::plot(time_units, snn.membrane_potentials[1]); */
        char buffer[50];
        sprintf(buffer, "Neuron %zu", i);
        plt::named_plot(buffer, time_units, snn.membrane_potentials[i]);
    }
    plt::legend();
    plt::subplot(2, 1, 2);
    plt::ylim(-0.1, 1.1);
    plt::title("Neuron 0 Spike Train");
    plt::plot(time_units, snn.spikes[0], "-o");
    plt::show();

    /* for (memory_index i = 0; i < num_neurons; i++) { */
    /*     /1* print_vector(snn.spikes[i]); *1/ */
    /*     char buffer[25]; */
    /*     sprintf(buffer, "Neuron: %zu", i); */
    /*     plt::named_plot(buffer, time_units, snn.spikes[i]); */
    /* } */
    /* plt::xlim(-10, static_cast<s32>(time_unit_limit)); */
    /* plt::legend(); */
    /* plt::title("Neuron Spikes"); */
    /* plt::legend(); */
    /* plt::show(); */
#endif

    return 0;
}
