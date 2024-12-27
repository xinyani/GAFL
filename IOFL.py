import torch
import pygad.torchga
import pygad
import numpy
import pandas as pd
import time

#服务器端收集来自各个客户端的模型测试结果来引导适应度函数更新
def fitness_func(ga_instanse, solution, sol_idx):
    global data_inputs, data_outputs, torch_ga, model, loss_function
    val_inputs=data_inputs
    val_outputs=data_outputs
    # 将验证数据分成10批,模拟10客户端
    batch_size = len(val_inputs) // 10
    total_loss = 0.0
    num_batches = 10
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size if i != num_batches - 1 else len(val_inputs)
        batch_inputs = val_inputs[start_idx:end_idx]
        batch_outputs = val_outputs[start_idx:end_idx]
        # 使用当前解决方案（权重）进行预测
        predictions = pygad.torchga.predict(model=model, solution=solution, data=batch_inputs)
        # 计算损失
        batch_loss = loss_function(predictions, batch_outputs)
        # 累加损失
        total_loss += batch_loss.detach().numpy()
    # 计算平均损失
    average_loss = total_loss / num_batches
    # 返回平均损失的倒数作为适应度值（因为我们希望损失越小越好）
    solution_fitness = 1.0 / average_loss if average_loss != 0 else float('-inf')  # 避免除以0的情况
    return solution_fitness
aa=[]
def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
    aa.append(ga_instance.best_solution()[1])

# Build the PyTorch model using the functional API.
input_layer = torch.nn.Linear(784, 200)
relu_layer = torch.nn.ReLU()
dense_layer = torch.nn.Linear(200, 50)
dense_layer2 = torch.nn.Linear(50, 10)
output_layer = torch.nn.Softmax(1)

model = torch.nn.Sequential(input_layer,
                            relu_layer,
                            dense_layer,
                            relu_layer,
                            dense_layer2,
                            # relu_layer,
                            output_layer)
print(model)
# Create an instance of the pygad.torchga.TorchGA class to build the initial population.5
torch_ga = pygad.torchga.TorchGA(model=model,
                                 num_solutions=20)

loss_function = torch.nn.CrossEntropyLoss()


df = pd.read_csv("mnist_train.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
data_inputs = torch.from_numpy(X/255).float()
data_outputs = torch.from_numpy(y).long()
num_generations = 1000 # Number of generations.20
num_parents_mating = 5 # Number of solutions to be selected as parents in the mating pool.
initial_population = torch_ga.population_weights # Initial population of network weights.

# Create an instance of the pygad.GA class
ga_instance = pygad.GA(num_generations=num_generations, 
                       num_parents_mating=num_parents_mating, 
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       on_generation=callback_generation)
print(ga_instance)
# Start the genetic algorithm evolution.
a=time.time()
ga_instance.run()
b=time.time()
print((b-a)/3600)
# After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
ga_instance.plot_fitness(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

predictions = pygad.torchga.predict(model=model, 
                                    solution=solution, 
                                    data=data_inputs)
# print("Predictions : \n", predictions)

# Calculate the crossentropy loss of the trained model.
print("Crossentropy : ", loss_function(predictions, data_outputs).detach().numpy())

# Calculate the classification accuracy for the trained model.
accuracy = torch.true_divide(torch.sum(torch.max(predictions, axis=1).indices == data_outputs), len(data_outputs))
print("Accuracy : ", accuracy.detach().numpy())

df = pd.read_csv("mnist_test.csv")
XX = df.iloc[:, :-1].values
yy = df.iloc[:, -1].values
data_input = torch.from_numpy(XX/255).float()
data_output = torch.from_numpy(yy).long()


predictions = pygad.torchga.predict(model=model,
                                    solution=solution,
                                    data=data_input)
# print("Predictions : \n", predictions)

# Calculate the crossentropy loss of the trained model.
print("TEST_Crossentropy : ", loss_function(predictions, data_output).detach().numpy())

# Calculate the classification accuracy for the trained model.
accuracy = torch.true_divide(torch.sum(torch.max(predictions, axis=1).indices == data_output), len(data_output))
print("TEST_Accuracy : ", accuracy.detach().numpy())
print(aa)