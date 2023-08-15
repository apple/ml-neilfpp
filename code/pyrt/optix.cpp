//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2020 Apple Inc. All Rights Reserved.
//
#include "SampleRenderer.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "3rdParty/stb_image_write.h"

/*! main entry point to this example - initially optix, print hello
world, then exit */
extern "C" int main(int ac, char **av)
{
try {
    std::vector<TriangleMesh> model(2);
    // 100x100 thin ground plane
    model[0].color = vec3f(0.f, 1.f, 0.f);
    model[0].addCube(vec3f(0.f,-1.5f, 0.f),vec3f(10.f,.1f,10.f));
    // a unit cube centered on top of that
    model[1].color = vec3f(0.f,1.f,1.f);
    model[1].addCube(vec3f(0.f,0.f,0.f),vec3f(2.f,2.f,2.f));

    Camera camera = { /*from*/vec3f(-10.f,2.f,-12.f),
                    /* at */vec3f(0.f,0.f,0.f),
                    /* up */vec3f(0.f,-1.f,0.f) };

    SampleRenderer sample(model);

    const vec2i fbSize(vec2i(1200,1024));
    sample.resize(fbSize);

    // sample.setCamera(camera);
    vec3f position  = camera.from;
    vec3f direction = normalize(camera.at-camera.from);
    const float cosFovy = 0.66f;
    const float aspect = fbSize.x / float(fbSize.y);
    vec3f horizontal = cosFovy * aspect * normalize(cross(direction, camera.up));
    vec3f vertical = cosFovy * normalize(cross(horizontal, direction));
    Ray ray;
    for (int iy=0; iy<fbSize.y; iy++) {
        for (int ix=0; ix<fbSize.x; ix++) {
            // normalized screen plane position, in [0,1]^2
            const vec2f screen(vec2f(ix+.5f,iy+.5f) / vec2f(fbSize));
            // generate ray direction
            vec3f rayDir = normalize(direction + (screen.x - 0.5f) * horizontal + (screen.y - 0.5f) * vertical);
            ray.origin.emplace_back(camera.from);
            ray.dir.emplace_back(rayDir);
        }
    }
    sample.setRaySBT(ray);

    sample.render();

    std::vector<float> pixels(fbSize.x*fbSize.y*6);
    sample.downloadPixels(pixels.data());

    std::vector<uint32_t> rgba(fbSize.x*fbSize.y);
    for (int i=0; i<fbSize.x*fbSize.y; i++) {
        const int r = int(255.99f*pixels[6*i]);
        const int g = int(255.99f*pixels[6*i+1]);
        const int b = int(255.99f*pixels[6*i+2]);

        // convert to 32-bit rgba value (we explicitly set alpha to 0xff
        // to make stb_image_write happy ...
        rgba[i] = 0xff000000
          | (r<<0) | (g<<8) | (b<<16);
    }

    const std::string fileName = "osc_example2.png";
    stbi_write_png(fileName.c_str(),fbSize.x,fbSize.y,4,
                    rgba.data(),fbSize.x*sizeof(uint32_t));
    std::cout << GDT_TERMINAL_GREEN
            << std::endl
            << "Image rendered, and saved to " << fileName << " ... done." << std::endl
            << GDT_TERMINAL_DEFAULT
            << std::endl;
} catch (std::runtime_error& e) {
    std::cout << GDT_TERMINAL_RED << "FATAL ERROR: " << e.what()
            << GDT_TERMINAL_DEFAULT << std::endl;
    exit(1);
}
return 0;
}
