import time
import pygame
import numpy as np
from PIL import Image
import gymnasium as gym
import ale_py
import threading

gym.register_envs(ale_py)

class HumanPlayMode:
    
    def __init__(self, time_limit_minutes=10, window_size=None):
        self.time_limit_minutes = time_limit_minutes
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
        
        pygame.init()
        self.display = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Pac-Man Human Play Mode")
        self.clock = pygame.time.Clock()
        
        # Create environment
        self.env = gym.make("ALE/Pacman-v5", render_mode="rgb_array")
        
        # Action mapping for ALE Pacman's 5-action space
        self.action_map = {
            pygame.K_UP: 1,     # UP
            pygame.K_RIGHT: 2,  # RIGHT
            pygame.K_DOWN: 4,   # DOWN
            pygame.K_LEFT: 3,   # LEFT
            pygame.K_SPACE: 0,  # NOOP
        }
        
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
    
    def start_timer(self):
        def timer_function():
            elapsed = 0
            while elapsed < self.time_limit_minutes * 60 and not self.time_expired:
                time.sleep(0.05)  # Check every 50ms
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
        
        step_text = f"Steps: {self.step_count}"
        self.draw_text(step_text, self.font_small, self.WHITE, (10, self.window_size[1] - 30))
        
        controls_text = "Use arrow keys to control, P to pause/resume"
        controls_width = self.font_small.size(controls_text)[0]
        controls_x = (self.window_size[0] - controls_width) // 2
        self.draw_text(controls_text, self.font_small, self.GREEN, (controls_x, self.window_size[1] - 30))
    
    def draw_countdown(self, count):
        # Render the current game state as background
        image = self.env.render()
        image = Image.fromarray(image, 'RGB')
        
        # Resize image to fit window
        image = image.resize(self.window_size, Image.Resampling.LANCZOS)
        
        # Convert to pygame surface
        mode, size, data = image.mode, image.size, image.tobytes()
        pygame_image = pygame.image.fromstring(data, size, mode)
        
        # Display game as background
        self.display.blit(pygame_image, (0, 0))
        
        # Semi-transparent overlay to dim the background
        overlay = pygame.Surface(self.window_size)
        overlay.set_alpha(180)  # More transparent to show game background
        overlay.fill(self.BLACK)
        self.display.blit(overlay, (0, 0))
        
        # Countdown text
        count_text = str(count)
        self.draw_text(count_text, self.font_large, self.YELLOW, 
                      (self.window_size[0]//2, self.window_size[1]//2), center=True)
        
        # Instructions
        if count == 3:
            instruction_text = "Get ready to play!"
        elif count == 2:
            instruction_text = "Collect dots and avoid ghosts!"
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
    
    def show_start_screen(self):
        self.display.fill(self.BLACK)
        
        # Title
        title_text = "Pac-Man Human Play Mode"
        self.draw_text(title_text, self.font_large, self.YELLOW,
                      (self.window_size[0]//2, 100), center=True)
        
        # Instructions
        instructions = [
            f"You will play Pac-Man for {self.time_limit_minutes} minutes",
            "Try to get the highest score possible!",
            "Your score will be saved and compared to others",
            "The highest score will receive a prize!",
            "",
            "You have unlimited lives - keep playing until time runs out!",
            "Use arrow keys to control, P to pause/resume",
            "",
            "Press any key to start..."
        ]
        
        for i, instruction in enumerate(instructions):
            color = self.YELLOW if i < 4 else self.WHITE
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
            'most_common_action': most_common_action
        }
    
    def show_end_screen(self, statistics):
        self.display.fill(self.BLACK)
        
        title_text = "Game Complete!"
        self.draw_text(title_text, self.font_large, self.YELLOW,
                      (self.window_size[0]//2, self.window_size[1]//2 - 100), center=True)
        
        results = [
            f"Final Score: {statistics['total_reward']}",
            "",
            "Press any key to exit..."
        ]
        
        for i, result in enumerate(results):
            color = self.YELLOW if i < 2 else self.WHITE
            y_pos = self.window_size[1]//2 - 50 + i * 30
            self.draw_text(result, self.font_medium, color,
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
    
    def run(self):
        print("Starting Pac-Man Human Play Mode")
        print("=" * 50)
        
        # Show start screen
        if not self.show_start_screen():
            return None
        
        # Reset environment
        obs, info = self.env.reset()
        print(f"Game started! Initial info: {info}")
        
        self.show_countdown()
        
        self.start_timer()
        
        # Main game loop
        running = True
        while running:
            # Update elapsed time for display (excluding pause time)
            if self.start_time:
                current_time = time.time()
                if self.paused and self.pause_start_time:
                    self.elapsed_time = self.pause_start_time - self.start_time - self.total_pause_time
                else:
                    # Normal calculation excluding total pause time
                    self.elapsed_time = current_time - self.start_time - self.total_pause_time
            
            # Check if time expired
            if self.time_expired:
                break
            
            # Handle events
            action = 0  # Default to NOOP
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
                    elif not self.paused and event.key in self.action_map:
                        action = self.action_map[event.key]
            
            if not running:
                break
            
            # If paused, skip game logic and continue loop
            if self.paused:
                self.clock.tick(60)
                continue
            
            # Take step in environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Update statistics
            self.total_reward += reward
            self.step_count += 1
            self.actions_taken.append(action)
            

            
            # Render game
            image = self.env.render()
            image = Image.fromarray(image, 'RGB')
            
            # Resize image to fit window
            image = image.resize(self.window_size, Image.Resampling.LANCZOS)
            
            # Convert to pygame surface
            mode, size, data = image.mode, image.size, image.tobytes()
            pygame_image = pygame.image.fromstring(data, size, mode)
            
            # Display game
            self.display.blit(pygame_image, (0, 0))
            
            # Draw overlay information
            self.draw_game_info()
            
            pygame.display.update()
            
            # Print progress every 100 steps
            if self.step_count % 100 == 0:
                time_remaining = max(0, self.time_limit_minutes * 60 - self.elapsed_time)
                print(f"Step {self.step_count}: Score={self.total_reward}, "
                      f"Time remaining: {int(time_remaining//60):02d}:{int(time_remaining%60):02d}")
            
            # Check if episode ended
            if terminated or truncated:
                print(f"Episode ended after {self.step_count} steps - continuing with unlimited lives!")
                # Reset environment to continue playing with unlimited lives
                obs, info = self.env.reset()

                # Continue the game loop
                continue
            
            target_fps = 20  # Adjust this value to control game speed
            self.clock.tick(target_fps)
        
        # Stop timer
        self.stop_timer()
        
        # Get final statistics
        statistics = self.get_game_statistics()
        
        # Show end screen
        self.show_end_screen(statistics)
        
        # Cleanup
        self.env.close()
        pygame.quit()
        
        print(f"\nGame completed!")
        print(f"Final Score: {statistics['total_reward']}")
        print(f"Total Steps: {statistics['step_count']}")
        print(f"Time Played: {self.elapsed_time:.1f} seconds")
        
        return statistics

def main():
    print("Pac-Man Human Play Mode - Study Phase 1")
    print("=" * 50)
    
    # Get time limit from user
    try:
        time_limit = input("Enter time limit in minutes (default 10): ").strip()
        time_limit = int(time_limit) if time_limit else 10
    except ValueError:
        time_limit = 10
    
    game = HumanPlayMode(time_limit_minutes=time_limit)
    statistics = game.run()
    
    if statistics:
        print("\nFinal Statistics:")
        for key, value in statistics.items():
            print(f"  {key}: {value}")
    
    print("\nThank you for participating!")

if __name__ == "__main__":
    main()
