import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Create a simple model
model = nn.Linear(10, 1)

# Initialize optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# Train for a few steps
for epoch in range(3):
    # Dummy forward and backward pass
    output = model(torch.randn(5, 10))
    loss = output.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Update scheduler with validation loss
    val_loss = 0.9 - epoch * 0.1  # Simulated decreasing validation loss
    scheduler.step(val_loss)

# Print the optimizer state
print("OPTIMIZER STATE:")
optimizer_state = optimizer.state_dict()
for key, value in optimizer_state.items():
    if key != 'state':
        print(f"{key}: {value}")
    else:
        print("state (contains buffers for each parameter):")
        # Print just one parameter's state as an example
        param_id = list(optimizer_state['state'].keys())[0]
        print(f"  Parameter {param_id} state:")
        for k, v in optimizer_state['state'][param_id].items():
            print(f"    {k}: {v}")

# Print the scheduler state
print("\nSCHEDULER STATE:")
scheduler_state = scheduler.state_dict()
for key, value in scheduler_state.items():
    print(f"{key}: {value}")