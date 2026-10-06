#import "AppDelegate.h"
#import <MetalKit/MetalKit.h>
#import "Renderer.h"

@interface AppDelegate ()
@property (nonatomic, strong) NSWindow *window;
@property (nonatomic, strong) Renderer *renderer;
@end

@implementation AppDelegate

- (void)applicationDidFinishLaunching:(NSNotification *)notification
{
    // 1. Создаём окно
    NSRect frame = NSMakeRect(100, 100, 1200, 800);

    self.window = [[NSWindow alloc]
        initWithContentRect:frame
        styleMask:(NSWindowStyleMaskTitled |
                   NSWindowStyleMaskClosable |
                   NSWindowStyleMaskResizable)
        backing:NSBackingStoreBuffered
        defer:NO];

    self.window.title = @"CV Lab";

    // 2. Metal device
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();

    if (!device)
    {
        NSLog(@"Metal is not supported");
        [NSApp terminate:nil];
        return;
    }

    // 3. Создаём MTKView
    MTKView *metalView = [[MTKView alloc] initWithFrame:self.window.contentView.bounds
                                                 device:device];

    metalView.autoresizingMask =
        NSViewWidthSizable | NSViewHeightSizable;

    metalView.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
    metalView.clearColor = MTLClearColorMake(0.1, 0.1, 0.1, 1.0);

    // 4. Создаём Renderer
    self.renderer = [[Renderer alloc] initWithView:metalView];

    // 5. Самое важное:
    metalView.delegate = self.renderer;

    // 6. Добавляем MTKView в окно
    self.window.contentView = metalView;

    // 7. Показываем окно
    [self.window makeKeyAndOrderFront:nil];

    [NSApp activateIgnoringOtherApps:YES];
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender
{
    return YES;
}

@end
