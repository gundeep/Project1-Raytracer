// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "sceneStructs.h"
#include <cutil_math.h>
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"
#include <vector>


void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 

//LOOK: This function demonstrates how to use thrust for random number generation on the GPU!
//Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y){
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}




//TODO: IMPLEMENT THIS FUNCTION
//Function that does the initial raycast from the camera
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov){
  
	ray r;
  r.origin = eye;
 
 
  float degToRad = PI/180.0;

		float radToDeg = 180.0/PI;

		float H = (float)(resolution[0] / 2);

		float V = (float)(resolution[1] / 2);

		//printf("%f", H);

		float camDistFromScreen = (float)(V / tan(fov[1] * degToRad));
		//printf(" %f " , camDistFromScreen);

		// Get the horizontal view in degrees

		float fieldViewX = (float)atan((H) / camDistFromScreen);

		fieldViewX *= radToDeg;

		//Find the vector C (Camera Direction)

		float multiplier = camDistFromScreen / sqrt(pow(view[0],2) + pow(view[1],2) + pow(view[2],2));

		glm::vec3 camVector = glm::vec3(view[0] * multiplier,view[1] * multiplier, view[2] * multiplier);

		glm::vec3 midpointScreen =glm::vec3(eye[0] + camVector.x, eye[1] +camVector.y, eye[2] + camVector.z) ;

	// Calculate NDC values for the coordinates
		
		float sx = (float)(1.0 * x / resolution[0]);

		float sy = (float)(1.0 * y / resolution[1]);

		//printf(" %f ", sx);
		
				//		 Calculate the point coordinates in 3D space

		glm::vec3 pointOnScreen;

		pointOnScreen.x = ((2 * sx) - 1) *H;

		pointOnScreen.y = ((2 * sy) - 1) *V;

		pointOnScreen.z = 0;

		glm::vec3 color= glm::vec3 (1,0,0);

		glm::vec3 P =midpointScreen + pointOnScreen;

		glm::vec3 Eye= glm::vec3(eye[0],eye[1],eye[2]);

		glm::vec3 direction= (P -Eye) / sqrt( pow(P.x-Eye.x,2) + pow(P.y - Eye.y,2) + pow(P.z - Eye.z,2));
					
		r.direction = direction;

		r.origin=eye;
	
		return r;
					
}

//Kernel that blacks out a given image buffer
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      image[index] = glm::vec3(0,0,0);
    }
}

//Kernel that writes the image to the OpenGL PBO directly. 
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
 
  int index = x + (y * resolution.x);
  
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;      
      color.x = image[index].x*255.0;
      color.y = image[index].y*255.0;
      color.z = image[index].z*255.0;

      if(color.x>255){
        color.x = 255;
      }
	  
      if(color.y>255){
        color.y = 255;
      }

      if(color.z>255){
        color.z = 255;
      }
      
      // Each thread writes one pixel location in the texture (textel)
	   y=resolution.y-y;
	   x=resolution.x-x;
	   index = x + (y * resolution.x);
      PBOpos[index].w = 0;
      PBOpos[index].x = color.x;     
      PBOpos[index].y = color.y;
      PBOpos[index].z = color.z;
  }
}

//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void raytraceRay(material* mats, glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors, 
                            staticGeom* geoms, int numberOfGeoms){

  glm::vec3 light_pos=glm::vec3(0,3,10);
  glm::vec3 light_col=glm::vec3(1.0,1.0,1.0);
  int anti_alias_loop=0;
  

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  while(anti_alias_loop<=4)
  {
	  
	  anti_alias_loop++;
	  int tempX=x;
	  int tempY=y;


	  if( anti_alias_loop==4)
	  {
		  
		  tempX+0.5;
	  }
	  else if( anti_alias_loop==1)
	  {
		  
		  tempY+0.5;

	  }

	  else if (anti_alias_loop==2)
	  {
		  tempX-0,5;
	  }

	  else if(anti_alias_loop==3);
	  {
		  tempY -0.5;

	  }

  glm::vec3 intersectionPoint= glm::vec3(0,0,0);
  glm::vec3 spec_color;

  colors[index]=glm::vec3(0,0,0);
  
  staticGeom* geometry=geoms;
  //material* mat=mats;

  glm::vec3 POI;
  
  if((x<=resolution.x && y<=resolution.y))
  {
	  ray ray_for_scene= raycastFromCameraKernel( resolution,  time,  tempX,  tempY, cam.position,  cam.view,  cam.up,  cam.fov);
	  int no_of_iterations=0;
	  
	 while(no_of_iterations <rayDepth )
	  {
		  float tmin=-1.0;
		  no_of_iterations++;
  
		  int mat_id;
		  glm::vec3 normal= glm::vec3(1,1,1);
	
	//Intersection test one!
		  for (int i=0; i<numberOfGeoms;++i)
		  {
	 
			if(geometry[i].type==GEOMTYPE::SPHERE)
			{
	
				float sphere_intersection_test=sphereIntersectionTest( geometry[i], ray_for_scene, intersectionPoint, normal);
				if(sphere_intersection_test>0)
				{
					if(tmin<0 || sphere_intersection_test<tmin)
					{
				
						tmin=sphere_intersection_test;
						//normal=tempnormal;
						mat_id=geometry[i].materialid;
						POI = intersectionPoint;
						
					}
			
					//int mat_id=geometry[i].materialid ;
					//colors[index]= mat[mat_id].color;
				}
		
			}
			else if(geometry[i].type==GEOMTYPE::CUBE)
			{
		
				float cube_intersection_test= boxIntersectionTest( geometry[i] , ray_for_scene, intersectionPoint, normal);
				if(cube_intersection_test>0)
				{
					if(tmin<0 || cube_intersection_test<tmin)
					{
			
						tmin=cube_intersection_test;
						//normal=tempnormal;
						mat_id=geometry[i].materialid;
						POI = intersectionPoint;
					}
					//int mat_id=geometry[i].materialid ;
					//colors[index]= mat[mat_id].color;
				}
		
			}
		  }
	
			  float tLight = -1;
			  ray lightRay;
			  lightRay.origin = light_pos;
			  lightRay.direction = POI - light_pos;
			  float dist = glm::length(lightRay.direction);
			  lightRay.direction = glm::normalize(lightRay.direction);
			  glm::vec3 lightIntersectPoint, lightNormal;
			  bool blocked = false;
			  if (tmin < 0)
			  {
				  colors[index]=glm::vec3(0,0,0);  // apply background color
			  }
			  

			  else
				  {
					  
					for (int i=0; i<numberOfGeoms;++i)
						{
							if (geoms[i].type == SPHERE)
							{
								float tLightTemp = sphereIntersectionTest(geoms[i], lightRay, lightIntersectPoint, lightNormal);
								//if (tLightTemp > 0 && (tLight < 0 || tLightTemp < tLight))
								if (tLightTemp > 0 && tLightTemp < (dist-0.01))
								{
									tLight = tLightTemp;
									blocked = true;
									//normal=lightNormal;
								}
							}

							if (geoms[i].type == CUBE)
							{
								float tLightTemp = boxIntersectionTest(geoms[i], lightRay, lightIntersectPoint, lightNormal);
								//if (tLightTemp > 0 && (tLight < 0 || tLightTemp < tLight))
								if (tLightTemp > 0 && tLightTemp < (dist-0.01))
								{
									tLight = tLightTemp;
									blocked = true;
									//normal=lightNormal;
								}
							}
						}

						//if (tLight > 0 && tLight < (dist - EPSILON))
						if (blocked)
						{
							// light is blocked
							colors[index]= glm::vec3(0,0,0);
						}
						else
						{
							if(no_of_iterations==0)
							{
							normal= glm::normalize(normal);
							lightRay.direction=glm::normalize(lightRay.direction);
							glm::vec3 diffuse= abs(glm::dot(lightRay.direction,normal))*mats[mat_id].color;
							glm::vec3 refelectedrayspec=glm::normalize(lightRay.direction -(normal+normal)*(glm::dot(lightRay.direction,normal)));

							float specdot= abs(glm::dot(refelectedrayspec,glm::normalize(POI-cam.position)));
							float specularity=0.2 * pow(specdot,mats[mat_id].specularExponent);
							colors[index]+= diffuse*light_col + light_col*specularity; //+(specularity/3 *light_col);
						//  colors[index]=  mats[mat_id].color;
							if(mats[mat_id].hasReflective >0 )
							  {

									normal=glm::normalize(normal);
									ray_for_scene.direction=glm::normalize(ray_for_scene.direction);
									glm::vec3 reflectedRay;
					
									reflectedRay= ray_for_scene.direction- (normal+normal)*(glm::dot(ray_for_scene.direction,normal));
									reflectedRay = glm::normalize(reflectedRay);

								    spec_color=mats[mat_id].hasReflective * mats[mat_id].specularColor;
									spec_color=glm::clamp(spec_color,0.0,1.0);

									ray_for_scene.origin=POI;
									ray_for_scene.direction=reflectedRay;
							  }
					  
							}

							else
							{
								normal= glm::normalize(normal);
							lightRay.direction=glm::normalize(lightRay.direction);
							glm::vec3 diffuse= (abs(glm::dot(lightRay.direction,normal))*mats[mat_id].color);
							glm::vec3 refelectedrayspec=glm::normalize(lightRay.direction -(normal+normal)*(glm::dot(lightRay.direction,normal)));

							float specdot= abs(glm::dot(refelectedrayspec,glm::normalize(POI-cam.position)));
							float specularity=0.2 * pow(specdot,mats[mat_id].specularExponent);
							colors[index]+= diffuse*light_col + light_col*specularity; //+(specularity/3 *light_col);

							colors[index].x=colors[index].x/2;
							colors[index].y=colors[index].y/2;
							colors[index].z=colors[index].z/2;

						//  colors[index]=  mats[mat_id].color;
							if(mats[mat_id].hasReflective >0 )
							  {
									normal=glm::normalize(normal);
									ray_for_scene.direction=glm::normalize(ray_for_scene.direction);
									glm::vec3 reflectedRay;
					
									reflectedRay= ray_for_scene.direction- (normal+normal)*(glm::dot(ray_for_scene.direction,normal));
									reflectedRay = glm::normalize(reflectedRay);

								    spec_color=mats[mat_id].hasReflective * mats[mat_id].specularColor;
									spec_color=glm::clamp(spec_color,0.0,1.0);

									ray_for_scene.origin=POI;
									ray_for_scene.direction=reflectedRay;

								}
							}

						}
						
					}


			}
		}
	
   }
   colors[index]=colors[index]/1.0f;
   colors[index]=glm::clamp(colors[index],0,1.0);
}


//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
  
  int traceDepth =2; //determines how many bounces the raytracer traces

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
  //send image to GPU
  glm::vec3* cudaimage = NULL;
  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);

  // send materials to GPU
 material* material_List= new material[numberOfMaterials];
 for (int i=0; i<numberOfMaterials;i++)
 {	
	
	 material mat;
	 mat.color=materials[i].color;
	 mat.specularColor=materials[i].specularColor;
	 mat.specularExponent=materials[i].specularExponent;
	 mat.hasReflective=materials[i].hasReflective;
	 mat.hasRefractive=materials[i].hasRefractive;
	 mat.indexOfRefraction=materials[i].indexOfRefraction;
	 mat.hasScatter=materials[i].hasScatter;
	 mat.absorptionCoefficient=materials[i].absorptionCoefficient;
	 mat.reducedScatterCoefficient=materials[i].reducedScatterCoefficient;
	 mat.emittance=materials[i].emittance;
	 material_List[i]= mat;
 }

 //
	material* cudamaterials= NULL;
	cudaMalloc((void**)&cudamaterials, numberOfMaterials*sizeof(material));
    cudaMemcpy( cudamaterials, material_List, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);

  //package geometry and materials and sent to GPU
  staticGeom* geomList = new staticGeom[numberOfGeoms];
  for(int i=0; i<numberOfGeoms; i++){
    staticGeom newStaticGeom;
    newStaticGeom.type = geoms[i].type;
    newStaticGeom.materialid = geoms[i].materialid;
    newStaticGeom.translation = geoms[i].translations[frame];
    newStaticGeom.rotation = geoms[i].rotations[frame];
    newStaticGeom.scale = geoms[i].scales[frame];
    newStaticGeom.transform = geoms[i].transforms[frame];
    newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
	newStaticGeom.tranposeTranform=geoms[i].tranposeTranforms[frame];
    geomList[i] = newStaticGeom;
  }

  staticGeom* cudageoms = NULL;
  cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
  cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);
  
  //package textures

  //package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;
  
 
  //kernel launches
  raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(cudamaterials , renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms);

  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);

  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  //free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree(cudamaterials);
  cudaFree( cudageoms );
  delete geomList;

  // make certain the kernel has completed 
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}
