use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

pub struct Network{
    layers: Vec<Layer>,
}

struct Layer{
    neurons: Vec<Neuron>
}

pub struct LayerTopology {
    pub neurons: usize,
}

struct Neuron {
    bias: f32,
    weights: Vec<f32>
}

impl Network {
    
    pub fn random( rng: &mut dyn rand::RngCore,layers: &[LayerTopology]) -> Self {
        assert!(layers.len() > 1);
        let layers = layers
        .windows(2)
        .map(|layers| {
            Layer::random(rng,layers[0].neurons, layers[1].neurons)
        })
        .collect();

        Self { layers }
    }
    
    fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.layers
            .iter()
            .fold(inputs, |inputs, layer| layer.propagate(inputs))
    }
}

impl Layer {
    
    pub fn random( rng: &mut dyn rand::RngCore,input_neurons: usize, output_neurons: usize) -> Self {
        let neurons = (0..output_neurons)
            .map(|_| Neuron::random(rng,input_neurons))
            .collect();
    
        Self { neurons }
    }

    fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.neurons
        .iter()
        .map(|neuron| neuron.propagate(&inputs))
        .collect()
    }
}

impl Neuron {

    pub fn random(
        rng: &mut dyn rand::RngCore,
        output_size: usize,
    ) -> Self {
            let bias = rng.gen_range(-1.0..=1.0);
    
        let weights = (0..output_size)
            .map(|_| rng.gen_range(-1.0..=1.0))
            .collect();
    
        Self { bias, weights }
    }

    fn propagate(&self, inputs: &[f32]) -> f32 {
        let output = inputs
            .iter()
            .zip(&self.weights)
            .map(|(input, weight)| input * weight)
            .sum::<f32>();

        (self.bias + output).max(0.0)
    }
}

#[test]
fn test() {
    let neuron = Neuron {
        bias: 0.5,
        weights: vec![-0.3, 0.8],
    };

    // Ensures `.max()` (our ReLU) works:
    approx::assert_relative_eq!(
        neuron.propagate(&[-10.0, -10.0]),
        0.0,
    );

    // `0.5` and `1.0` chosen by a fair dice roll:
    approx::assert_relative_eq!(
        neuron.propagate(&[0.5, 1.0]),
        (-0.3 * 0.5) + (0.8 * 1.0) + 0.5,
    );

    // We could've written `1.15` right away, but showing the entire
    // formula makes our intentions clearer
}