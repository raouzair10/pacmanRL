import time
import os
import pygame
import numpy as np
from PIL import Image
import gymnasium as gym
import ale_py
import threading
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper

gym.register_envs(ale_py)

class AgentPlayMode:
    
    def __init__(self, model_path="ppo_pacman.zip", time_limit_minutes=10, countdown_seconds=5, freeze_mode_first=True, window_size=None):
        self.model_path = model_path
        self.time_limit_minutes = time_limit_minutes
        self.countdown_seconds = countdown_seconds
        self.freeze_mode_first = freeze_mode_first
        if window_size is None:
            pygame.init()
            info = pygame.display.Info()
            self.window_size = (int(info.current_w * 0.6), int(info.current_h * 0.8))
        else:
            self.window_size = window_size
        self.start_time = None
        self.elapsed_time = 0
        self.pause_start_time = None
        self.total_pause_time = 0
        self.game_timer = None
        self.time_expired = False
        self.paused = False
        self.total_reward = 0
        self.step_count = 0
        self.actions_taken = []
        
        # Advice system variables
        self.advice_frequency = 50  # Ask for advice every 50 steps
        self.waiting_for_advice = False
        self.human_advice_count = 0
        self.agent_action_count = 0
        self.current_advice_mode = "freeze" if freeze_mode_first else "countdown"  # Current mode: "freeze" or "countdown"
        self.mode_switch_time = (self.time_limit_minutes * 60) / 2  # Switch modes at halfway point
        
        pygame.init()
        self.display = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Pac-Man Agent Play Mode")
        self.clock = pygame.time.Clock()
        
        self.env = self.create_pacman_env()
        
        try:
            self.agent = PPO.load(model_path)
            print(f"Successfully loaded agent from {model_path}")
        except Exception as e:
            print(f"Error loading agent from {model_path}: {e}")
            print("Please make sure the model file exists and is valid.")
            raise
        
        # Font setup
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.YELLOW = (255, 255, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.ORANGE = (255, 165, 0)
        
        # Action mapping for human advice
        self.action_map = {
            pygame.K_UP: 1,     # UP
            pygame.K_RIGHT: 2,  # RIGHT
            pygame.K_DOWN: 4,   # DOWN
            pygame.K_LEFT: 3,   # LEFT
            pygame.K_SPACE: 0,  # NOOP
        }
    
    def create_pacman_env(self):
        env = gym.make("ALE/Pacman-v5", render_mode="rgb_array")
        
        env = AtariWrapper(
            env,
            frame_skip=4,        # Skip 4 frames
            terminal_on_life_loss=False,  # Don't end episode on life loss
            clip_reward=True     # Clip rewards to [-1, 1]
        )
        
        return env
    
    def start_timer(self):
        def timer_function():
            elapsed = 0
            while elapsed < self.time_limit_minutes * 60 and not self.time_expired:
                time.sleep(0.1)  # Check every 100ms
                if not self.paused:  # Only count time when not paused
                    elapsed += 0.1
            if elapsed >= self.time_limit_minutes * 60:
                self.time_expired = True
                print(f"\nTime limit reached! ({self.time_limit_minutes} minutes)")
        
        self.game_timer = threading.Thread(target=timer_function, daemon=True)
        self.game_timer.start()
        self.start_time = time.time()
    
    def stop_timer(self):
        self.time_expired = True
        if self.game_timer and self.game_timer.is_alive():
            pass
        
    def draw_text(self, text, font, color, position, center=False):
        text_surface = font.render(text, True, color)
        if center:
            text_rect = text_surface.get_rect(center=position)
        else:
            text_rect = text_surface.get_rect(topleft=position)
        self.display.blit(text_surface, text_rect)
    
    def draw_game_info(self):
        time_remaining = max(0, self.time_limit_minutes * 60 - self.elapsed_time)
        minutes = int(time_remaining // 60)
        seconds = int(time_remaining % 60)
        time_text = f"Time: {minutes:02d}:{seconds:02d}"
        self.draw_text(time_text, self.font_medium, self.WHITE, (10, 10))
        
        score_text = f"Score: {self.total_reward}"
        score_width = self.font_medium.size(score_text)[0]
        score_x = self.window_size[0] - score_width - 10
        self.draw_text(score_text, self.font_medium, self.YELLOW, (score_x, 10))
        
        mode_text = f"Mode: {self.current_advice_mode.upper()}"
        mode_width = self.font_small.size(mode_text)[0]
        mode_x = (self.window_size[0] - mode_width) // 2
        self.draw_text(mode_text, self.font_small, self.BLUE, (mode_x, 10))
        
        step_text = f"Steps: {self.step_count}"
        self.draw_text(step_text, self.font_small, self.WHITE, (10, self.window_size[1] - 30))
        
        controls_text = "AI Agent Playing - Press P to pause/resume, Esc to exit"
        controls_width = self.font_small.size(controls_text)[0]
        controls_x = (self.window_size[0] - controls_width) // 2
        self.draw_text(controls_text, self.font_small, self.ORANGE, (controls_x, self.window_size[1] - 30))
    
    def draw_countdown(self, count):
        # Render the current game state as background
        image = self.env.render()
        image = Image.fromarray(image, 'RGB')
        
        image = image.resize(self.window_size, Image.Resampling.LANCZOS)
        
        mode, size, data = image.mode, image.size, image.tobytes()
        pygame_image = pygame.image.fromstring(data, size, mode)
        
        self.display.blit(pygame_image, (0, 0))
        
        # Semi-transparent overlay to dim the background
        overlay = pygame.Surface(self.window_size)
        overlay.set_alpha(180)
        overlay.fill(self.BLACK)
        self.display.blit(overlay, (0, 0))
        
        # Countdown text
        count_text = str(count)
        self.draw_text(count_text, self.font_large, self.YELLOW, 
                      (self.window_size[0]//2, self.window_size[1]//2), center=True)
        
        # Instructions
        if count == 3:
            instruction_text = "AI Agent getting ready!"
        elif count == 2:
            instruction_text = "Watch the AI play Pac-Man!"
        elif count == 1:
            instruction_text = "Go!"
        else:
            instruction_text = ""
        
        if instruction_text:
            self.draw_text(instruction_text, self.font_medium, self.WHITE,
                          (self.window_size[0]//2, self.window_size[1]//2 + 60), center=True)
        
        pygame.display.update()
    
    def draw_pause_screen(self):
        # Semi-transparent overlay
        overlay = pygame.Surface(self.window_size)
        overlay.set_alpha(150)
        overlay.fill(self.BLACK)
        self.display.blit(overlay, (0, 0))
        
        # Pause text
        pause_text = "PAUSED"
        self.draw_text(pause_text, self.font_large, self.YELLOW,
                      (self.window_size[0]//2, self.window_size[1]//2 - 50), center=True)
        
        # Instructions
        instruction_text = "Press P to resume, Esc to exit"
        self.draw_text(instruction_text, self.font_medium, self.WHITE,
                      (self.window_size[0]//2, self.window_size[1]//2 + 50), center=True)
        
        pygame.display.update()
    
    def show_mode_switch_screen(self):
        image = self.env.render()
        image = Image.fromarray(image, 'RGB')
        
        image = image.resize(self.window_size, Image.Resampling.LANCZOS)
        
        mode, size, data = image.mode, image.size, image.tobytes()
        pygame_image = pygame.image.fromstring(data, size, mode)
        
        self.display.blit(pygame_image, (0, 0))
        
        overlay = pygame.Surface(self.window_size)
        overlay.set_alpha(180)
        overlay.fill(self.BLACK)
        self.display.blit(overlay, (0, 0))
        
        switch_text = "MODE CHANGE!"
        self.draw_text(switch_text, self.font_large, self.YELLOW,
                      (self.window_size[0]//2, self.window_size[1]//2 - 80), center=True)
        
        mode_text = f"Switching to {self.current_advice_mode.upper()} MODE"
        self.draw_text(mode_text, self.font_medium, self.GREEN,
                      (self.window_size[0]//2, self.window_size[1]//2 - 30), center=True)
        
        if self.current_advice_mode == "freeze":
            description = "Freeze mode: Wait indefinitely for human advice"
        else:
            description = f"Countdown mode: Wait {self.countdown_seconds} seconds for advice"
        
        self.draw_text(description, self.font_small, self.WHITE,
                      (self.window_size[0]//2, self.window_size[1]//2 + 10), center=True)
        
        instruction_text = "Press SPACE or ENTER to continue"
        self.draw_text(instruction_text, self.font_medium, self.ORANGE,
                      (self.window_size[0]//2, self.window_size[1]//2 + 60), center=True)
        
        pygame.display.update()
    
    def draw_advice_screen(self, countdown_time=None):
        # First render the current game state as background
        image = self.env.render()
        image = Image.fromarray(image, 'RGB')
        
        image = image.resize(self.window_size, Image.Resampling.LANCZOS)
        
        mode, size, data = image.mode, image.size, image.tobytes()
        pygame_image = pygame.image.fromstring(data, size, mode)
        
        self.display.blit(pygame_image, (0, 0))
        
        # Semi-transparent overlay
        overlay = pygame.Surface(self.window_size)
        overlay.set_alpha(150)
        overlay.fill(self.BLACK)
        self.display.blit(overlay, (0, 0))
        
        # Advice request text in the center
        advice_text = "HUMAN ADVICE NEEDED!"
        self.draw_text(advice_text, self.font_large, self.YELLOW,
                      (self.window_size[0]//2, self.window_size[1]//2 - 50), center=True)
        
        # Instructions in the center
        instructions = [
            "Use arrow keys to give advice:",
            "UP     RIGHT     DOWN     LEFT",
        ]
        
        # Add countdown info if in countdown mode
        if countdown_time is not None:
            instructions.append(f"Time remaining: {int(countdown_time)}s")
        
        for i, instruction in enumerate(instructions):
            y_pos = self.window_size[1]//2 + 20 + i * 30
            self.draw_text(instruction, self.font_medium, self.YELLOW,
                          (self.window_size[0]//2, y_pos), center=True)
        
        pygame.display.update()
    
    def show_start_screen(self):
        self.display.fill(self.BLACK)
        
        # Title
        title_text = "Pac-Man AI Agent Play Mode"
        self.draw_text(title_text, self.font_large, self.YELLOW,
                      (self.window_size[0]//2, 100), center=True)
        
        # Instructions
        instructions = [
            f"AI Agent will play Pac-Man for {self.time_limit_minutes} minutes.",
            "The agent was trained using reinforcement learning.",
            "See how well it performs compared to humans.",
            "",
            f"Freeze mode waits indefinitely for advice.",
            f"Countdown mode waits {self.countdown_seconds} seconds for advice.",
            "Use arrow keys to guide the agent when prompted.",
            "",
            "Press any key to start..."
        ]
        
        for i, instruction in enumerate(instructions):
            if i < 3:
                color = self.YELLOW
            elif i == 4 or i == 5:
                color = self.BLUE
            elif i == 6:
                color = self.GREEN
            elif i == 7:
                color = self.ORANGE
            else:
                color = self.WHITE
            y_pos = 200 + i * 30
            self.draw_text(instruction, self.font_medium, color,
                          (self.window_size[0]//2, y_pos), center=True)
        
        pygame.display.update()
        
        # Wait for key press
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    waiting = False
                    break
            self.clock.tick(60)
        
        return True
    
    def show_countdown(self):
        for count in [3, 2, 1]:
            self.draw_countdown(count)
            time.sleep(1)
    
    def get_game_statistics(self):
        if not self.actions_taken:
            return {}
        
        action_counts = [self.actions_taken.count(i) for i in range(5)]
        most_common_action = max(set(self.actions_taken), key=self.actions_taken.count)
        
        return {
            'total_reward': self.total_reward,
            'step_count': self.step_count,
            'elapsed_time_seconds': self.elapsed_time,
            'average_reward_per_step': self.total_reward / max(self.step_count, 1),
            'actions_per_second': self.step_count / max(self.elapsed_time, 1),
            'action_distribution': action_counts,
            'most_common_action': most_common_action,
            'human_advice_count': self.human_advice_count,
            'agent_action_count': self.agent_action_count,
            'advice_ratio': self.human_advice_count / max(self.step_count, 1)
        }
    
    def show_end_screen(self, statistics):
        self.display.fill(self.BLACK)
        
        title_text = "AI Agent Game Complete!"
        self.draw_text(title_text, self.font_large, self.YELLOW,
                      (self.window_size[0]//2, self.window_size[1]//2 - 100), center=True)
        
        results = [
            f"Final Score: {statistics['total_reward']}",
            "Thank you for playing!",
            "",
            "Press any key to exit..."
        ]
        
        for i, result in enumerate(results):
            y_pos = self.window_size[1]//2 - 60 + i * 30
            self.draw_text(result, self.font_medium, self.YELLOW,
                          (self.window_size[0]//2, y_pos), center=True)
        
        pygame.display.update()
        
        # Wait for key press
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    waiting = False
                    break
            self.clock.tick(60)
        
        return True
    
    def request_human_advice(self):
        self.waiting_for_advice = True
        
        if self.current_advice_mode == "freeze":
            return self.request_human_advice_freeze()
        else:
            return self.request_human_advice_countdown()
    
    def request_human_advice_freeze(self):
        self.draw_advice_screen()
        
        # Wait for human input indefinitely
        waiting = True
        action = 0  # Default to NOOP
        
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None  # Exit signal
                elif event.type == pygame.KEYDOWN:
                    if event.key in self.action_map:
                        action = self.action_map[event.key]
                        waiting = False
                        break
                    elif event.key == pygame.K_ESCAPE:
                        return None
            
            self.clock.tick(60)
        
        self.waiting_for_advice = False
        self.human_advice_count += 1
        return action
    
    def request_human_advice_countdown(self):
        start_time = time.time()
        action = None
        
        while time.time() - start_time < self.countdown_seconds:
            remaining_time = self.countdown_seconds - (time.time() - start_time)
            self.draw_advice_screen(countdown_time=remaining_time)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                elif event.type == pygame.KEYDOWN:
                    if event.key in self.action_map:
                        action = self.action_map[event.key]
                        self.human_advice_count += 1
                        print(f"Human advised action: {action}")
                        break
                    elif event.key == pygame.K_ESCAPE:
                        return None
            
            if action is not None:
                break
            
            self.clock.tick(60)
        
        self.waiting_for_advice = False
        
        if action is None:
            # No advice given, use agent's action
            print(f"No advice given within {self.countdown_seconds} seconds, using agent's action")
            return "agent_action"
        
        return action
    
    def run(self):
        print("Starting Pac-Man AI Agent Play Mode")
        print("=" * 50)
        
        if not self.show_start_screen():
            return None
        
        obs, info = self.env.reset()
        print(f"Game started! Initial info: {info}")
        
        self.show_countdown()
        
        self.start_timer()
        
        # Main game loop
        running = True
        while running:
            if self.start_time:
                current_time = time.time()
                if self.paused and self.pause_start_time:
                    self.elapsed_time = self.pause_start_time - self.start_time - self.total_pause_time
                else:
                    self.elapsed_time = current_time - self.start_time - self.total_pause_time
            
            if self.time_expired:
                break
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        # Toggle pause state
                        self.paused = not self.paused
                        if self.paused:
                            # Start tracking pause time
                            self.pause_start_time = time.time()
                            self.draw_pause_screen()
                        else:
                            # End pause and add to total pause time
                            if self.pause_start_time:
                                self.total_pause_time += time.time() - self.pause_start_time
                                self.pause_start_time = None
                    elif event.key == pygame.K_ESCAPE and self.paused:
                        running = False
                        break
            
            if not running:
                break
            
            if self.paused:
                self.clock.tick(60)
                continue
            
            # if it's time to switch modes (at halfway point)
            if self.elapsed_time >= self.mode_switch_time and self.current_advice_mode == ("freeze" if self.freeze_mode_first else "countdown"):
                # Pause the game and timer for mode switch
                self.paused = True
                self.pause_start_time = time.time()
                
                # Switch to the other mode
                self.current_advice_mode = "countdown" if self.freeze_mode_first else "freeze"
                print(f"\nMode switched to {self.current_advice_mode.upper()} mode at {self.elapsed_time:.1f} seconds!")
                
                # Show mode switch screen
                self.show_mode_switch_screen()
                
                # Wait for user to continue
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                            break
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE or event.key == pygame.K_RETURN:
                                waiting = False
                                break
                            elif event.key == pygame.K_ESCAPE:
                                running = False
                                break
                    
                    if not running:
                        break
                    self.clock.tick(60)
                
                if not running:
                    break
                
                # Resume the game
                self.paused = False
                if self.pause_start_time:
                    self.total_pause_time += time.time() - self.pause_start_time
                    self.pause_start_time = None
            
            # Check if it's time to ask for human advice
            if self.step_count > 0 and self.step_count % self.advice_frequency == 0:
                print(f"\nStep {self.step_count}: Requesting human advice in {self.current_advice_mode} mode...")
                advice_action = self.request_human_advice()
                
                if advice_action is None:
                    running = False
                    break
                
                if advice_action == "agent_action":
                    # No advice given in countdown mode, use agent's action
                    action, _ = self.agent.predict(obs, deterministic=True)
                    action = int(action)
                    self.agent_action_count += 1
                    print(f"Using agent's action: {action}")
                else:
                    # Human gave advice
                    action = int(advice_action)
                    print(f"Human advised action: {action}")
            else:
                # Get action from the trained agent
                action, _ = self.agent.predict(obs, deterministic=True)
                action = int(action)  
                self.agent_action_count += 1
                if self.step_count % 100 == 0:
                    print(f"Agent taking action: {action}")
            
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            self.total_reward += reward
            self.step_count += 1
            self.actions_taken.append(action)
            

            
            # Render game
            image = self.env.render()
            image = Image.fromarray(image, 'RGB')
            
            image = image.resize(self.window_size, Image.Resampling.LANCZOS)
            
            mode, size, data = image.mode, image.size, image.tobytes()
            pygame_image = pygame.image.fromstring(data, size, mode)
            
            self.display.blit(pygame_image, (0, 0))
            
            self.draw_game_info()
            
            pygame.display.update()
            
            # Print progress every 100 steps
            if self.step_count % 100 == 0:
                time_remaining = max(0, self.time_limit_minutes * 60 - self.elapsed_time)
                print(f"Step {self.step_count}: Score={self.total_reward}, "
                      f"Time remaining: {int(time_remaining//60):02d}:{int(time_remaining%60):02d}")
            
            if terminated or truncated:
                print(f"Episode ended after {self.step_count} steps - continuing with unlimited lives!")
                obs, info = self.env.reset()

                continue
            
            target_fps = 3  # Adjust this value to control game speed
            self.clock.tick(target_fps)
        
        self.stop_timer()
        
        statistics = self.get_game_statistics()
        
        self.show_end_screen(statistics)
        
        self.env.close()
        pygame.quit()
        
        print(f"\nAI Agent game completed!")
        print(f"Final Score: {statistics['total_reward']}")
        print(f"Total Steps: {statistics['step_count']}")
        print(f"Time Played: {self.elapsed_time:.1f} seconds")
        
        return statistics

def main():
    print("Pac-Man AI Agent Play Mode with Human Advice")
    print("=" * 50)
    
    # Get time limit from user
    try:
        time_limit = input("Enter time limit in minutes (default 10): ").strip()
        time_limit = int(time_limit) if time_limit else 10
    except ValueError:
        time_limit = 10
    
    # Get countdown time from user
    try:
        countdown_time = input(f"Enter countdown time in seconds (default 5): ").strip()
        countdown_time = int(countdown_time) if countdown_time else 5
    except ValueError:
        countdown_time = 5
    
    # Get which mode goes first
    while True:
        mode_choice = input("Which mode should go first? (1 for Freeze, 2 for Countdown): ").strip()
        if mode_choice == "1":
            freeze_mode_first = True
            break
        elif mode_choice == "2":
            freeze_mode_first = False
            break
        else:
            print("Please enter 1 for Freeze mode first or 2 for Countdown mode first.")
    
    model_path = "ppo_pacman.zip"
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please make sure you have trained the agent first using train_agent.py")
        return
    
    game = AgentPlayMode(
        model_path=model_path, 
        time_limit_minutes=time_limit,
        countdown_seconds=countdown_time,
        freeze_mode_first=freeze_mode_first
    )
    statistics = game.run()
    
    if statistics:
        print("\nFinal Statistics:")
        for key, value in statistics.items():
            print(f"  {key}: {value}")
   
if __name__ == "__main__":
    
    main()
