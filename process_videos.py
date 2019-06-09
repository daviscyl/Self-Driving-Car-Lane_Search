from utils import Lane_Finder
lane_finder = Lane_Finder()

video_files = ['videos/project_video.mp4',
               'videos/challenge_video.mp4',
               'videos/harder_challenge_video.mp4']

print('\nFiles to be processed:\n'+'\n'.join(video_files)+'\n')

for video in video_files:
    print('processing: ' + video + ' ... ', end='')
    lane_finder.process_video(video)
    print('done.')
