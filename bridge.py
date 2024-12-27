import pygame
import vgamepad

# Initialize Pygame and VGamepad
pygame.init()
pygame.joystick.init()

# Create a virtual gamepad
gamepad = vgamepad.VX360Gamepad()

# Helper function to map values
def map_value(value,invert, in_min=-1.0, in_max=1.0, out_min=-32768, out_max=32767):
    """Map a value from one range to another."""
    if invert == True:
        value = -value
    return int((value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

# Process input from two controllers
def process_input(controller1, controller2):
    try:
        while True:
            pygame.event.pump()  # Process input events

            # Map left stick of Controller 1 to Virtual Gamepad
            left_x = map_value(controller1.get_axis(0),invert = False)  # Left stick X
            left_y = map_value(controller1.get_axis(1),invert = True)  # Left stick Y
            gamepad.left_joystick(x_value=left_x, y_value=left_y)

            # Map right stick of Controller 2 to Virtual Gamepad
            right_x = map_value(controller2.get_axis(0),invert = False)  # Right stick X
            right_y = map_value(controller2.get_axis(1),invert = True)  # Right stick Y
            gamepad.right_joystick(x_value=right_x, y_value=right_y)


            # Map triggers from Controller 2
            left_trigger = int((controller2.get_axis(2) + 1) * 127.5)  # Scale [-1, 1] to [0, 255]
            right_trigger = int((controller2.get_axis(5) + 1) * 127.5)  # Scale [-1, 1] to [0, 255]
            gamepad.left_trigger(value=left_trigger)
            gamepad.right_trigger(value=right_trigger)

            # Map buttons from Controller 1
            if controller1.get_button(0):  # A button
                gamepad.press_button(vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_A)
            else:
                gamepad.release_button(vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_A)

            if controller1.get_button(1):  # B button
                gamepad.press_button(vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_B)
            else:
                gamepad.release_button(vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_B)

            if controller1.get_button(2):  # X button
                gamepad.press_button(vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_X)
            else:
                gamepad.release_button(vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_X)

            if controller1.get_button(3):  # Y button
                gamepad.press_button(vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_Y)
            else:
                gamepad.release_button(vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_Y)

            # Map buttons from Controller 2
            if controller2.get_button(4):  # Left bumper
                gamepad.press_button(vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)
            else:
                gamepad.release_button(vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)

            if controller2.get_button(5):  # Right bumper
                gamepad.press_button(vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)
            else:
                gamepad.release_button(vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)

            # Send updated inputs to the virtual gamepad
            gamepad.update()
    except KeyboardInterrupt:
        print("Exiting...")
        pygame.quit()

def main():
    # Check number of controllers
    if pygame.joystick.get_count() < 2:
        print("Two controllers are required.")
        return

    # Initialize controllers
    controller1 = pygame.joystick.Joystick(0)
    controller2 = pygame.joystick.Joystick(1)
    controller1.init()
    controller2.init()

    print(f"Controller 1: {controller1.get_name()}")
    print(f"Controller 2: {controller2.get_name()}")

    process_input(controller1, controller2)

if __name__ == "__main__":
    main()
