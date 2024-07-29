import matplotlib.pyplot as plt

def read_results(method_name):
    try:
        with open(f"{method_name}_results.txt", "r") as file:
            lines = file.readlines()
            accuracy = float(lines[0].strip().split(": ")[1])
            process_time = float(lines[1].strip().split(": ")[1])
        return accuracy, process_time
    except FileNotFoundError:
        print(f"File for {method_name} not found.")
        return None, None

# Example method names
methods = ['pyzbar', 'pyboof', 'OpenCV', 'YOLO']

accuracies = []
process_times = []

for method in methods:
    accuracy, process_time = read_results(method)
    if accuracy is not None and process_time is not None:
        accuracies.append(accuracy)
        process_times.append(process_time)

# Plotting results
plt.figure(figsize=(12, 6))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.bar(methods, accuracies, color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Methods')
plt.ylabel('Accuracy')
plt.title('Accuracy of Different Methods')
plt.ylim(0, 1)

# Plot processing time
plt.subplot(1, 2, 2)
plt.bar(methods, process_times, color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Methods')
plt.ylabel('Processing Time (ms)')
plt.title('Processing Time of Different Methods')

plt.tight_layout()
plt.savefig('results.png')
plt.show()
